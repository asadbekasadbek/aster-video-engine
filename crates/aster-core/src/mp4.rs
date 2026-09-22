//! Rust streaming adapter to an explicitly selected native FFmpeg executable.
//! No downloads and no inference; callers supply ordered RGB8 frames.
use crate::video::VideoSpec;
use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
    process::{Child, Command, Stdio},
};

pub struct RgbFrames<W: Write> {
    output: W,
    bytes: usize,
    frames: usize,
    next: usize,
    failed: bool,
}

impl<W: Write> RgbFrames<W> {
    fn new(output: W, spec: VideoSpec) -> io::Result<Self> {
        spec.frame_bytes()?;
        let bytes = spec
            .width
            .checked_mul(spec.height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| io::Error::other("RGB size overflow"))?;
        Ok(Self {
            output,
            bytes,
            frames: spec.frames,
            next: 0,
            failed: false,
        })
    }
    pub fn write_frame(&mut self, index: usize, rgb: &[u8]) -> io::Result<()> {
        if self.failed || index != self.next || index >= self.frames || rgb.len() != self.bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid RGB frame order, size or failed stream",
            ));
        }
        self.failed = true;
        self.output.write_all(rgb)?;
        self.failed = false;
        self.next += 1;
        Ok(())
    }
    fn finish(mut self) -> io::Result<()> {
        if self.failed || self.next != self.frames {
            return Err(io::Error::other("incomplete RGB stream"));
        }
        self.output.flush()
    }
}

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Writes fragmented H.264 MP4 so FFmpeg can use a non-seekable stdout handle.
/// Existing outputs are never overwritten. Failures can leave a partial output.
/// The encoder's own memory is outside core buffer accounting.
pub fn encode(
    ffmpeg: &Path,
    output: &Path,
    spec: VideoSpec,
    produce: impl FnOnce(&mut RgbFrames<std::process::ChildStdin>) -> io::Result<()>,
) -> io::Result<()> {
    spec.frame_bytes()?;
    if output.extension().and_then(|s| s.to_str()) != Some("mp4") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "output must end in .mp4",
        ));
    }
    // Require a local executable selected by the user, not shell/PATH interpolation.
    if !ffmpeg.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "FFmpeg executable not found",
        ));
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut command = Command::new(ffmpeg);
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
        ])
        .arg(format!("{}x{}", spec.width, spec.height))
        .arg("-framerate")
        .arg(format!("{}/{}", spec.fps_num, spec.fps_den))
        .args([
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            "-movflags",
            "frag_keyframe+empty_moov",
            "-f",
            "mp4",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::from(file))
        .stderr(Stdio::inherit());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let mut child = Process(command.spawn()?);
    let stdin = child
        .0
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("encoder stdin unavailable"))?;
    let mut writer = RgbFrames::new(stdin, spec)?;
    produce(&mut writer)?;
    writer.finish()?; // Closes stdin before wait, including buffered EOF.
    let status = child.0.wait()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "FFmpeg encoding failed: {status}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frame_contract_and_partial_stream() {
        let spec = VideoSpec {
            width: 2,
            height: 2,
            frames: 2,
            fps_num: 24,
            fps_den: 1,
        };
        let mut w = RgbFrames::new(Vec::new(), spec).unwrap();
        assert!(w.write_frame(1, &[0; 12]).is_err());
        assert!(w.write_frame(0, &[0; 11]).is_err());
        w.write_frame(0, &[0; 12]).unwrap();
        assert!(w.finish().is_err());
        let mut storage = [0; 8];
        let mut w = RgbFrames::new(storage.as_mut_slice(), spec).unwrap();
        assert!(w.write_frame(0, &[0; 12]).is_err());
        assert!(w.write_frame(0, &[0; 12]).is_err());
    }
}
