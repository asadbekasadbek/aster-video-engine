//! Minimal uncompressed YUV4MPEG2 boundary; no external encoder or AI model.
use std::io::{self, Write};

#[derive(Clone, Copy, Debug)]
pub struct VideoSpec {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps_num: u32,
    pub fps_den: u32,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

impl VideoSpec {
    pub fn frame_bytes(self) -> io::Result<usize> {
        if self.width == 0
            || self.height == 0
            || self.frames == 0
            || !self.width.is_multiple_of(2)
            || !self.height.is_multiple_of(2)
            || self.fps_num == 0
            || self.fps_den == 0
        {
            return Err(invalid(
                "nonzero even dimensions, frames and frame rate required",
            ));
        }
        let pixels = self
            .width
            .checked_mul(self.height)
            .ok_or_else(|| invalid("shape overflow"))?;
        let bytes = pixels
            .checked_add(pixels / 2)
            .ok_or_else(|| invalid("frame size overflow"))?;
        let total = bytes
            .checked_add(6)
            .and_then(|n| n.checked_mul(self.frames))
            .ok_or_else(|| invalid("video size overflow"))?;
        if bytes > 64 * 1024 * 1024 || total > 1024 * 1024 * 1024 {
            return Err(invalid(
                "prototype video limit: 64 MiB/frame, 1 GiB frame payload plus markers",
            ));
        }
        Ok(bytes)
    }
}

/// Frame i has timestamp i * fps_den / fps_num seconds. Payload is planar 8-bit
/// YUV420 full range; no audio. The writer rejects out-of-order/incomplete streams.
pub struct Y4mWriter<W: Write> {
    output: W,
    spec: VideoSpec,
    bytes: usize,
    next_frame: usize,
    failed: bool,
}

impl<W: Write> Y4mWriter<W> {
    pub fn new(mut output: W, spec: VideoSpec) -> io::Result<Self> {
        let bytes = spec.frame_bytes()?;
        writeln!(
            output,
            "YUV4MPEG2 W{} H{} F{}:{} Ip A1:1 C420jpeg XCOLORRANGE=FULL",
            spec.width, spec.height, spec.fps_num, spec.fps_den
        )?;
        Ok(Self {
            output,
            spec,
            bytes,
            next_frame: 0,
            failed: false,
        })
    }

    pub fn write_frame(&mut self, index: usize, yuv: &[u8]) -> io::Result<()> {
        if self.failed {
            return Err(invalid("writer is poisoned after an I/O failure"));
        }
        if index != self.next_frame || index >= self.spec.frames || yuv.len() != self.bytes {
            return Err(invalid("unexpected frame index, count or payload length"));
        }
        self.failed = true;
        self.output.write_all(b"FRAME\n")?;
        self.output.write_all(yuv)?;
        self.failed = false;
        self.next_frame += 1;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<W> {
        if self.failed || self.next_frame != self.spec.frames {
            return Err(invalid("incomplete video"));
        }
        self.output.flush()?;
        Ok(self.output)
    }
}

/// Reproducible moving square for exercising media I/O; not model output.
pub fn write_demo(output: impl Write) -> io::Result<()> {
    let spec = VideoSpec {
        width: 320,
        height: 180,
        frames: 48,
        fps_num: 24,
        fps_den: 1,
    };
    let mut payload = vec![128_u8; spec.frame_bytes()?];
    let mut writer = Y4mWriter::new(output, spec)?;
    for frame in 0..spec.frames {
        let left = frame * (spec.width - 32) / (spec.frames - 1);
        for y in 0..spec.height {
            for x in 0..spec.width {
                payload[y * spec.width + x] =
                    if (left..left + 32).contains(&x) && (74..106).contains(&y) {
                        235
                    } else {
                        24
                    };
            }
        }
        writer.write_frame(frame, &payload)?;
    }
    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> VideoSpec {
        VideoSpec {
            width: 2,
            height: 2,
            frames: 2,
            fps_num: 24,
            fps_den: 1,
        }
    }

    #[test]
    fn exact_bytes_and_order() {
        let mut w = Y4mWriter::new(Vec::new(), spec()).unwrap();
        assert!(w.write_frame(1, &[0; 6]).is_err());
        assert!(w.write_frame(0, &[0; 5]).is_err());
        w.write_frame(0, &[0; 6]).unwrap();
        assert!(w.write_frame(0, &[0; 6]).is_err());
        w.write_frame(1, &[255; 6]).unwrap();
        assert!(w.write_frame(2, &[0; 6]).is_err());
        let result = w.finish().unwrap();
        let mut expected =
            b"YUV4MPEG2 W2 H2 F24:1 Ip A1:1 C420jpeg XCOLORRANGE=FULL\nFRAME\n".to_vec();
        expected.extend([0; 6]);
        expected.extend(b"FRAME\n");
        expected.extend([255; 6]);
        assert_eq!(result, expected);
    }

    #[test]
    fn rejects_incomplete_and_invalid_specs() {
        assert!(
            Y4mWriter::new(Vec::new(), spec())
                .unwrap()
                .finish()
                .is_err()
        );
        for s in [
            VideoSpec { width: 3, ..spec() },
            VideoSpec {
                fps_den: 0,
                ..spec()
            },
            VideoSpec {
                frames: usize::MAX,
                ..spec()
            },
            VideoSpec {
                width: usize::MAX - 1,
                ..spec()
            },
        ] {
            assert!(s.frame_bytes().is_err());
        }
    }

    #[test]
    fn io_failure_cannot_be_retried_as_success() {
        let header = b"YUV4MPEG2 W2 H2 F24:1 Ip A1:1 C420jpeg XCOLORRANGE=FULL\n";
        // One complete frame and only part of the next frame marker fit.
        let mut storage = vec![0_u8; header.len() + 12 + 3];
        let mut w = Y4mWriter::new(storage.as_mut_slice(), spec()).unwrap();
        w.write_frame(0, &[0; 6]).unwrap();
        assert!(w.write_frame(1, &[0; 6]).is_err());
        assert!(w.write_frame(1, &[0; 6]).is_err());
        assert!(w.finish().is_err());
    }
}
