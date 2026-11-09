use candle_core::{DType, Device, Tensor};
use candle_video::models::ltx_video::weight_format::KeyRemapper;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn streaming() -> Result<bool> {
    match std::env::var("ASTER_WEIGHT_MODE").as_deref() {
        Ok("resident") => Ok(false),
        Ok("stream") | Err(std::env::VarError::NotPresent) => Ok(true),
        _ => Err("ASTER_WEIGHT_MODE must be stream or resident".into()),
    }
}

pub struct BlockFile {
    file: std::sync::Mutex<File>,
    metadata: aster_core::model::Inspection,
}

impl BlockFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let metadata = aster_core::model::inspect(&mut file)?;
        Ok(Self {
            file: std::sync::Mutex::new(file),
            metadata,
        })
    }

    pub fn builder(
        &self,
        index: Option<usize>,
        device: &Device,
    ) -> Result<candle_nn::VarBuilder<'static>> {
        Ok(candle_nn::VarBuilder::from_tensors(
            self.tensors(index, device)?,
            DType::BF16,
            device,
        ))
    }

    pub fn tensors(
        &self,
        index: Option<usize>,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let scope = index.map_or(Scope::Shared, |i| {
            Scope::Block(format!("model.diffusion_model.transformer_blocks.{i}."))
        });
        let mut file = self.file.lock().map_err(|_| "checkpoint lock poisoned")?;
        load_from(
            &mut *file,
            &self.metadata,
            scope,
            DType::BF16,
            device,
            256 * 1024 * 1024,
        )
    }
}

enum Scope {
    Vae,
    Encoder,
    Decoder,
    Dit,
    Shared,
    Block(String),
}
impl Scope {
    fn accepts(&self, key: &str) -> bool {
        match self {
            Self::Vae => key.starts_with("vae."),
            Self::Encoder => {
                key.starts_with("vae.encoder.") || key.starts_with("vae.per_channel_statistics.")
            }
            Self::Decoder => {
                key.starts_with("vae.decoder.") || key.starts_with("vae.per_channel_statistics.")
            }
            Self::Dit => key.starts_with("model.diffusion_model."),
            Self::Shared => {
                Self::Dit.accepts(key)
                    && !key.starts_with("model.diffusion_model.transformer_blocks.")
            }
            Self::Block(prefix) => key.starts_with(prefix),
        }
    }
}

pub fn verify(path: &Path, size: u64, expected: &str) -> Result<()> {
    let mut file = File::open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() != size {
        return Err(format!("wrong file type or size: {}", path.display()).into());
    }
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    if format!("{:x}", hash.finalize()) != expected {
        return Err(format!("SHA-256 mismatch: {}", path.display()).into());
    }
    Ok(())
}

pub fn vae_half(
    path: &Path,
    encoder: bool,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    let mut file = File::open(path)?;
    let index = aster_core::model::inspect(&mut file)?;
    let scope = if encoder {
        Scope::Encoder
    } else {
        Scope::Decoder
    };
    let map = load_from(
        &mut file,
        &index,
        scope,
        DType::BF16,
        device,
        crate::inference::WEIGHT_LIMIT,
    )?;
    Ok(candle_nn::VarBuilder::from_tensors(
        map,
        DType::BF16,
        device,
    ))
}

/// Reads ONLY selected component payloads, one tensor at a time. Retained GPU
/// weight bytes and a 256 MiB single-tensor bound are checked before allocation.
/// Backend activation/workspace overhead is additional, not enforced by this cap.
pub fn load(
    path: &Path,
    vae: bool,
    dtype: DType,
    device: &Device,
    limit: usize,
) -> Result<HashMap<String, Tensor>> {
    let mut file = File::open(path)?;
    let inspection = aster_core::model::inspect(&mut file)?;
    load_from(
        &mut file,
        &inspection,
        if vae { Scope::Vae } else { Scope::Dit },
        dtype,
        device,
        limit,
    )
}

fn load_from(
    reader: &mut (impl Read + Seek),
    inspection: &aster_core::model::Inspection,
    scope: Scope,
    dtype: DType,
    device: &Device,
    limit: usize,
) -> Result<HashMap<String, Tensor>> {
    let remapper = KeyRemapper::new();
    let mut total = 0_usize;
    let mut selected = Vec::new();
    for key in inspection.metadata.offset_keys() {
        if scope.accepts(&key) {
            let info = inspection
                .metadata
                .info(&key)
                .ok_or("missing tensor metadata")?;
            let input_dtype = match info.dtype {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::BF16 => DType::BF16,
                _ => return Err("unsupported weight dtype; no implicit FP8 conversion".into()),
            };
            let bytes = info.data_offsets.1 - info.data_offsets.0;
            if bytes > 256 * 1024 * 1024 {
                return Err("single weight tensor exceeds 256 MiB staging budget".into());
            }
            let retained = (bytes / input_dtype.size_in_bytes())
                .checked_mul(dtype.size_in_bytes())
                .ok_or("weight size overflow")?;
            total = total
                .checked_add(retained)
                .ok_or("weight budget overflow")?;
            if total > limit {
                return Err("component exceeds configured weight budget".into());
            }
            selected.push((key, input_dtype));
        }
    }
    if selected.is_empty() {
        return Err("selected model component is absent".into());
    }
    let mut tensors = HashMap::new();
    let mut residency = aster_core::residency::Manager::new([0, 512 * 1024 * 1024, limit]);
    for (id, (key, input_dtype)) in selected.into_iter().enumerate() {
        let info = inspection
            .metadata
            .info(&key)
            .ok_or("missing tensor metadata")?;
        let mut raw = vec![0; info.data_offsets.1 - info.data_offsets.0];
        reader.seek(SeekFrom::Start(
            8 + inspection.header_bytes + info.data_offsets.0 as u64,
        ))?;
        reader.read_exact(&mut raw)?;
        let retained = raw.len() / input_dtype.size_in_bytes() * dtype.size_in_bytes();
        let transfer = move || -> Result<Tensor> {
            if input_dtype == dtype {
                return Ok(Tensor::from_raw_buffer(&raw, dtype, &info.shape, device)?);
            }
            let source = Tensor::from_raw_buffer(&raw, input_dtype, &info.shape, &Device::Cpu)?;
            drop(raw);
            Ok(source.to_dtype(dtype)?.to_device(device)?)
        };
        let tensor = if device.is_cuda() {
            use aster_core::residency::{Resource, Tier};
            residency.register(Resource {
                id: id as u32,
                bytes: retained,
                tier: Tier::Ram,
            })?;
            residency.transfer_value(id as u32, Tier::Vram, 0, |_, _| -> Result<Tensor> {
                let destination = transfer()?;
                device.synchronize()?;
                Ok(destination)
            })?
        } else {
            transfer()?
        };
        let remapped = remapper.remap_key(&key);
        let clean = remapped
            .strip_prefix("vae.")
            .or_else(|| remapped.strip_prefix("model.diffusion_model."))
            .or_else(|| remapped.strip_prefix("transformer."))
            .unwrap_or(&remapped);
        if tensors.insert(clean.to_owned(), tensor).is_some() {
            return Err("duplicate remapped tensor key".into());
        }
    }
    println!(
        "stage={} weight_payload_bytes={total}",
        match scope {
            Scope::Vae => "vae_load",
            Scope::Encoder => "vae_encoder_load",
            Scope::Decoder => "vae_decoder_load",
            Scope::Dit => "dit_load",
            Scope::Shared => "dit_shared",
            Scope::Block(_) => "dit_block",
        }
    );
    if device.is_cuda() {
        println!(
            "residency_scope=component_weight_transfer reserved_peak_bytes={:?} resident_bytes={:?}",
            residency.reserved_peak(),
            residency.used()
        );
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disk_groups_partition_shared_and_indexed_weights() {
        let block = Scope::Block("model.diffusion_model.transformer_blocks.1.".into());
        let one = "model.diffusion_model.transformer_blocks.1.attn1.to_k.weight";
        let ten = "model.diffusion_model.transformer_blocks.10.attn1.to_k.weight";
        assert!(block.accepts(one));
        assert!(!block.accepts(ten));
        assert!(!Scope::Shared.accepts(one));
        assert!(Scope::Shared.accepts("model.diffusion_model.proj_out.bias"));
        assert!(!Scope::Shared.accepts("vae.decoder.weight"));
    }
    #[test]
    fn loads_only_selected_component_and_enforces_budget() {
        // VAE time embeddings must not match the transformer's broad upstream heuristic.
        let header = r#"{"model.diffusion_model.w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"vae.time_embed.w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(3_f32.to_le_bytes());
        bytes.extend(7_f32.to_le_bytes());
        let mut reader = std::io::Cursor::new(bytes);
        let metadata = aster_core::model::inspect(&mut reader).unwrap();
        let tensors = load_from(
            &mut reader,
            &metadata,
            Scope::Dit,
            DType::F32,
            &Device::Cpu,
            4,
        )
        .unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["w"].to_vec1::<f32>().unwrap(), [3.]);
        let tensors = load_from(
            &mut reader,
            &metadata,
            Scope::Vae,
            DType::F32,
            &Device::Cpu,
            4,
        )
        .unwrap();
        assert_eq!(tensors["time_embed.w"].to_vec1::<f32>().unwrap(), [7.]);
        assert!(
            load_from(
                &mut reader,
                &metadata,
                Scope::Dit,
                DType::F32,
                &Device::Cpu,
                3
            )
            .is_err()
        );
    }
}
