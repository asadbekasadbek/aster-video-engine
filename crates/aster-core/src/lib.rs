//! Bounded CPU reference pipeline. No model inference or GPU backend is implied.
pub mod memory;
pub mod model;
pub mod video;

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Invalid(&'static str),
    Overflow,
    Budget { required: usize, limit: usize },
    Allocation,
    Cancelled,
}

impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "invalid_input",
            Self::Overflow => "size_overflow",
            Self::Budget { .. } => "memory_budget",
            Self::Allocation => "allocation_failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Overflow => f.write_str("shape or work size overflow"),
            Self::Budget { required, limit } => {
                write!(
                    f,
                    "tensor buffers require {required} bytes; budget is {limit}"
                )
            }
            Self::Allocation => f.write_str("cannot allocate tensor buffers"),
            Self::Cancelled => f.write_str("job cancelled"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub frames: usize,
    pub seed: u64,
    /// Bounds tensor payload only; process/runtime overhead is measured separately.
    pub budget_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            channels: 4,
            frames: 24,
            seed: 42,
            budget_bytes: 64 * 1024 * 1024,
        }
    }
}

impl Config {
    pub fn validate(self) -> Result<(usize, usize), Error> {
        if [self.width, self.height, self.channels, self.frames].contains(&0) {
            return Err(Error::Invalid("dimensions and frames must be nonzero"));
        }
        let elements = self
            .width
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(self.channels))
            .ok_or(Error::Overflow)?;
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::Overflow)?;
        let work = elements.checked_mul(self.frames).ok_or(Error::Overflow)?;
        if bytes > self.budget_bytes {
            return Err(Error::Budget {
                required: bytes,
                limit: self.budget_bytes,
            });
        }
        if bytes > 1024 * 1024 * 1024 || work > 1_000_000_000 {
            return Err(Error::Invalid(
                "prototype limit: 1 GiB buffers and 1 billion element updates",
            ));
        }
        Ok((elements, bytes))
    }
}

#[derive(Debug)]
pub struct Report {
    pub checksum: u64,
    pub frames: usize,
    pub tensor_bytes: usize,
    pub elapsed: Duration,
}

fn buffer(elements: usize) -> Result<Vec<f32>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(elements)
        .map_err(|_| Error::Allocation)?;
    result.resize(elements, 0.0);
    Ok(result)
}

/// Streaming temporal recurrence: y[t,i] = .75*y[t-1,i] + .25*noise[t,i].
/// Only two CHW frames reside in memory. Consumer runs synchronously on each frame;
/// retaining copies in a consumer is outside this pipeline's memory budget.
/// Cancellation is checked before allocation and every 4096 element updates.
pub fn run(
    config: Config,
    cancelled: &AtomicBool,
    mut consume: impl FnMut(usize, &[f32]) -> Result<(), Error>,
) -> Result<Report, Error> {
    let (elements, tensor_bytes) = config.validate()?;
    let check_cancel = || {
        if cancelled.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    };
    check_cancel()?;
    let start = Instant::now();
    let mut previous = buffer(elements)?;
    let mut current = buffer(elements)?;
    let mut state = config.seed;
    let mut checksum = 0xcbf29ce484222325_u64;
    for frame in 0..config.frames {
        check_cancel()?;
        for i in 0..elements {
            if i % 4096 == 0 {
                check_cancel()?;
            }
            // Fully specified wrapping generator; zero is a valid seed.
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 40) as u32 as f32 / 16_777_216.0) * 2.0 - 1.0;
            current[i] = previous[i] * 0.75 + noise * 0.25;
            checksum ^= u64::from(current[i].to_bits());
            checksum = checksum.wrapping_mul(0x100000001b3);
        }
        check_cancel()?;
        consume(frame, &current)?;
        std::mem::swap(&mut previous, &mut current);
    }
    check_cancel()?;
    Ok(Report {
        checksum,
        frames: config.frames,
        tensor_bytes,
        elapsed: start.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_sensitive() {
        let stop = AtomicBool::new(false);
        let c = Config::default();
        let a = run(c, &stop, |_, _| Ok(())).unwrap();
        let b = run(c, &stop, |_, _| Ok(())).unwrap();
        let d = run(Config { seed: 43, ..c }, &stop, |_, _| Ok(())).unwrap();
        assert_eq!(a.checksum, b.checksum);
        assert_ne!(a.checksum, d.checksum);
    }

    #[test]
    fn independent_scalar_reference_and_bounds() {
        let c = Config {
            width: 1,
            height: 1,
            channels: 1,
            frames: 3,
            seed: 0,
            budget_bytes: 8,
        };
        let mut values = Vec::new();
        run(c, &AtomicBool::new(false), |_, x| {
            values.push(x[0]);
            Ok(())
        })
        .unwrap();
        // First RNG state is 1; noise is exactly -1, so output is -.25.
        assert_eq!(values[0], -0.25);
        let noise2 = ((6364136223846793006_u64 >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0;
        assert_eq!(values[1], -0.1875 + 0.25 * noise2);
        assert!(values.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn invalid_inputs_fail_before_consumer() {
        for (c, expected) in [
            (
                Config {
                    width: 0,
                    ..Config::default()
                },
                "invalid_input",
            ),
            (
                Config {
                    width: usize::MAX,
                    ..Config::default()
                },
                "size_overflow",
            ),
            (
                Config {
                    budget_bytes: 1,
                    ..Config::default()
                },
                "memory_budget",
            ),
            (
                Config {
                    frames: usize::MAX,
                    ..Config::default()
                },
                "size_overflow",
            ),
            (
                Config {
                    frames: 100_000,
                    ..Config::default()
                },
                "invalid_input",
            ),
        ] {
            let e = run(c, &AtomicBool::new(false), |_, _| {
                panic!("must not consume")
            })
            .unwrap_err();
            assert_eq!(e.code(), expected);
        }
    }

    #[test]
    fn cancellation_and_consumer_error_propagate() {
        let stop = AtomicBool::new(true);
        assert_eq!(
            run(Config::default(), &stop, |_, _| panic!()).unwrap_err(),
            Error::Cancelled
        );
        stop.store(false, Ordering::Relaxed);
        let mut count = 0;
        let e = run(Config::default(), &stop, |_, _| {
            count += 1;
            stop.store(true, Ordering::Relaxed);
            Ok(())
        })
        .unwrap_err();
        assert_eq!(e, Error::Cancelled);
        assert_eq!(count, 1);
        stop.store(false, Ordering::Relaxed);
        assert_eq!(
            run(Config::default(), &stop, |_, _| Err(Error::Invalid("sink"))).unwrap_err(),
            Error::Invalid("sink")
        );
    }

    #[test]
    fn memory_does_not_grow_with_frame_count() {
        let c = Config::default();
        assert_eq!(
            c.validate().unwrap().1,
            Config { frames: 240, ..c }.validate().unwrap().1
        );
        assert_eq!(c.validate().unwrap().1, 131_072);
    }
}
