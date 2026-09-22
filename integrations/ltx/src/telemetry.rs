use std::{
    process::Command,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

pub struct Monitor {
    stop: mpsc::Sender<()>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Monitor {
    pub fn start() -> Self {
        let (stop, receive) = mpsc::channel();
        let handle = thread::spawn(move || {
            let start = Instant::now();
            let mut baseline = None;
            let mut peak = 0_u64;
            let mut samples = 0_u64;
            let mut utilization = 0_f64;
            let mut failures = 0_u64;
            loop {
                let mut command = Command::new("nvidia-smi");
                command.args([
                    "--id=0",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]);
                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    command.creation_flags(0x08000000);
                }
                let observation = command
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .and_then(|o| {
                        let text = String::from_utf8(o.stdout).ok()?;
                        let (memory, utilization) = text.trim().split_once(',')?;
                        Some((
                            memory.trim().parse::<u64>().ok()?,
                            utilization.trim().parse::<f64>().ok()?,
                        ))
                    });
                if let Some((memory, busy)) = observation {
                    baseline.get_or_insert(memory);
                    peak = peak.max(memory);
                    samples += 1;
                    utilization += busy;
                } else {
                    failures += 1;
                }
                if receive.recv_timeout(Duration::from_millis(200))
                    != Err(mpsc::RecvTimeoutError::Timeout)
                {
                    break;
                }
            }
            if let Some(baseline) = baseline {
                println!(
                    "telemetry_scope=device_wide gpu_baseline_mib={baseline} gpu_sampled_peak_mib={peak} gpu_samples={samples} gpu_sample_errors={failures} gpu_mean_utilization_percent={} gpu_sampling_wait_ms=200 telemetry_elapsed_ms={} energy=unavailable",
                    utilization / samples as f64,
                    start.elapsed().as_millis()
                );
            } else {
                println!("gpu_peak_vram=unavailable gpu_sample_errors={failures}");
            }
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
