//! Optional CPU read-ahead and cross-step GPU weight caching for streamed DiT.
use crate::weights::{BlockFile, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::{
    collections::HashMap,
    io::Read,
    path::Path,
    sync::{Arc, Mutex, mpsc},
    thread,
};

type Builder = VarBuilder<'static>;
type Packet = std::result::Result<(usize, HashMap<String, Tensor>), String>;
const SWITCH: &str = "ASTER_PREFETCH";
const RAM: &str = "ASTER_PREFETCH_RAM_PERCENT";
const VRAM: &str = "ASTER_PREFETCH_VRAM_PERCENT";
const MIB: usize = 1024 * 1024;
// Same hard per-group payload limit as BlockFile.
const SMALL_BLOCK_LIMIT: usize = 256 * MIB;

fn file_values(contents: &str) -> HashMap<&str, &str> {
    contents
        .trim_start_matches('\u{feff}')
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split('#').next()?.split_once('=')?;
            let key = key.trim();
            matches!(key, SWITCH | RAM | VRAM | "ASTER_QUALITY_PROFILE")
                .then_some((key, value.trim()))
        })
        .collect()
}

fn setting(file: &HashMap<&str, &str>, name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(file.get(name).map(|s| s.to_string())),
        Err(e) => Err(e.into()),
    }
}

fn percentage(name: &str, value: Option<&str>, default: u8) -> Result<u8> {
    match value {
        Some(value) => value
            .parse::<u8>()
            .ok()
            .filter(|p| *p <= 100)
            .ok_or_else(|| format!("{name} must be an integer from 0 to 100").into()),
        None => Ok(default),
    }
}

fn allowance(free: usize, percent: u8, reserve: usize) -> usize {
    (free / 100 * usize::from(percent)).min(free.saturating_sub(reserve))
}

#[cfg(windows)]
fn free_ram() -> Result<usize> {
    #[repr(C)]
    #[derive(Default)]
    struct Status {
        length: u32,
        load: u32,
        total: u64,
        available: u64,
        page_total: u64,
        page_free: u64,
        virtual_total: u64,
        virtual_free: u64,
        extended: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut Status) -> i32;
    }
    let mut status = Status {
        length: size_of::<Status>() as u32,
        ..Default::default()
    };
    // SAFETY: full writable MEMORYSTATUSEX with its required size initialized.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(usize::try_from(status.available)?)
}
#[cfg(not(windows))]
fn free_ram() -> Result<usize> {
    Err("RAM prefetch measurement requires Windows; set ASTER_PREFETCH_RAM_PERCENT=0".into())
}

#[cfg(feature = "cuda")]
pub(crate) fn free_vram(device: &Device) -> Result<usize> {
    Ok(device
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .mem_get_info()?
        .0)
}
#[cfg(not(feature = "cuda"))]
pub(crate) fn free_vram(_: &Device) -> Result<usize> {
    Err("GPU caching requires a CUDA build".into())
}

fn dotenv() -> Result<String> {
    let contents = match std::fs::File::open(".env") {
        Ok(file) => {
            let mut text = String::new();
            file.take(65537).read_to_string(&mut text)?;
            if text.len() > 65536 {
                return Err("Aster .env exceeds 64 KiB".into());
            }
            text
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };
    Ok(contents)
}

pub fn quality_profile() -> Result<Option<String>> {
    setting(&file_values(&dotenv()?), "ASTER_QUALITY_PROFILE")
}

pub struct Stream {
    file: Arc<BlockFile>,
    queue: usize,
    cache_slots: usize,
    cache: Mutex<HashMap<usize, Builder>>,
}

impl Stream {
    pub fn open(path: &Path, device: &Device, layers: usize) -> Result<Self> {
        // Both the loader and the cache budget use this bound. A 13B block
        // contains 537,059,328 BF16 payload bytes, above the 2B limit.
        let slot = if layers == 48 {
            544 * MIB
        } else {
            SMALL_BLOCK_LIMIT
        };
        let contents = dotenv()?;
        let file = file_values(&contents);
        let enabled = match setting(&file, SWITCH)?.as_deref() {
            Some("1") => true,
            Some("0") | None => false,
            _ => return Err("ASTER_PREFETCH must be 0 or 1".into()),
        };
        let (queue, cache_slots) = if enabled {
            let ram = percentage(RAM, setting(&file, RAM)?.as_deref(), 10)?;
            let gpu = percentage(VRAM, setting(&file, VRAM)?.as_deref(), 25)?;
            let host = if ram == 0 {
                0
            } else {
                allowance(free_ram()?, ram, 1024 * MIB)
            };
            let vram = if gpu == 0 {
                0
            } else {
                allowance(free_vram(device)?, gpu, 6144 * MIB)
            };
            // In addition to the queue: a producer, a consuming block and
            // temporary raw/converted tensor buffers during CPU deserialization.
            let queue = (host / slot).saturating_sub(4).min(layers);
            let slots = (vram / slot).min(layers);
            println!(
                "prefetch=on ram_percent={ram} vram_percent={gpu} ram_budget_bytes={host} vram_budget_bytes={vram} queue_blocks={queue} gpu_cache_blocks={slots}"
            );
            (queue, slots)
        } else {
            (0, 0)
        };
        Ok(Self {
            file: Arc::new(BlockFile::open(path, slot)?),
            queue,
            cache_slots,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn shared(&self, device: &Device) -> Result<Builder> {
        self.file.builder(None, device)
    }

    pub fn start(&self, layers: usize, skip: &[usize]) -> Result<Option<Reader>> {
        if self.queue == 0 {
            return Ok(None);
        }
        let cache = self.cache.lock().map_err(|_| "weight cache poisoned")?;
        let indices: Vec<_> = (0..layers)
            .filter(|i| !skip.contains(i) && !cache.contains_key(i))
            .collect();
        if indices.is_empty() {
            return Ok(None);
        }
        let file = Arc::clone(&self.file);
        let (send, recv) = mpsc::sync_channel(self.queue);
        let worker = thread::Builder::new()
            .name("aster-weight-reader".into())
            .spawn(move || {
                for index in indices {
                    let packet = file
                        .tensors(Some(index), &Device::Cpu)
                        .map(|map| (index, map))
                        .map_err(|e| e.to_string());
                    let failed = packet.is_err();
                    if send.send(packet).is_err() || failed {
                        break;
                    }
                }
            })?;
        Ok(Some(Reader {
            recv: Some(recv),
            worker: Some(worker),
        }))
    }

    pub fn block(&self, index: usize, device: &Device, reader: Option<&Reader>) -> Result<Builder> {
        if index < self.cache_slots
            && let Some(vb) = self
                .cache
                .lock()
                .map_err(|_| "weight cache poisoned")?
                .get(&index)
        {
            return Ok(vb.clone());
        }
        let vb = if let Some(reader) = reader {
            let (received, map) = reader
                .recv
                .as_ref()
                .ok_or("closed weight queue")?
                .recv()??;
            if received != index {
                return Err("out-of-order prefetched block".into());
            }
            let gpu = map
                .into_iter()
                .map(|(name, cpu)| Ok((name, cpu.to_device(device)?)))
                .collect::<candle_core::Result<HashMap<_, _>>>()?;
            VarBuilder::from_tensors(gpu, DType::BF16, device)
        } else {
            self.file.builder(Some(index), device)?
        };
        if index < self.cache_slots {
            self.cache
                .lock()
                .map_err(|_| "weight cache poisoned")?
                .insert(index, vb.clone());
        }
        Ok(vb)
    }
}

pub struct Reader {
    recv: Option<mpsc::Receiver<Packet>>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Drop for Reader {
    fn drop(&mut self) {
        // Unblock a full sender on early return before waiting for its exit.
        drop(self.recv.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dropping_reader_releases_a_blocked_producer() {
        let (sender, receiver) = mpsc::sync_channel(0);
        let (done, observed) = mpsc::channel();
        let worker = thread::spawn(move || {
            let cancelled = sender.send(Ok((0, HashMap::new()))).is_err();
            done.send(cancelled).unwrap();
        });
        drop(Reader {
            recv: Some(receiver),
            worker: Some(worker),
        });
        assert!(observed.recv().unwrap());
    }
    #[test]
    fn reads_only_prefetch_keys_and_checks_percentages() {
        let v = file_values(
            "\u{feff}ASTER_PREFETCH=1\r\nIGNORED=yes\n ASTER_PREFETCH_RAM_PERCENT = 12 # comment\n",
        );
        assert_eq!(v.len(), 2);
        assert_eq!(v[SWITCH], "1");
        assert_eq!(percentage(RAM, Some(v[RAM]), 10).unwrap(), 12);
        assert_eq!(percentage(RAM, None, 10).unwrap(), 10);
        for invalid in ["-1", "101", "1.5", ""] {
            assert!(percentage(RAM, Some(invalid), 10).is_err());
        }
    }
    #[test]
    fn reads_default_quality_profile() {
        let options =
            file_values("# defaults\nASTER_QUALITY_PROFILE=reference13b-704-fullvae # portrait");
        assert_eq!(
            options.get("ASTER_QUALITY_PROFILE"),
            Some(&"reference13b-704-fullvae")
        );
    }

    #[test]
    fn allowances_preserve_headroom_and_respect_zero() {
        assert_eq!(allowance(1000, 25, 100), 250);
        assert_eq!(allowance(1000, 100, 400), 600);
        assert_eq!(allowance(100, 50, 400), 0);
        assert_eq!(allowance(1000, 0, 0), 0);
    }
}
