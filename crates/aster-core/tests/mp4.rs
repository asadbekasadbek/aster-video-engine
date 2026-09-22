use aster_core::{mp4, video::VideoSpec};
use std::{path::PathBuf, process::Command};

#[test]
#[ignore = "requires ASTER_TEST_FFMPEG and ASTER_TEST_FFPROBE pointing to installed executables"]
fn native_mp4_roundtrip() {
    let ffmpeg =
        PathBuf::from(std::env::var_os("ASTER_TEST_FFMPEG").expect("set ASTER_TEST_FFMPEG"));
    let ffprobe =
        PathBuf::from(std::env::var_os("ASTER_TEST_FFPROBE").expect("set ASTER_TEST_FFPROBE"));
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-artifacts");
    std::fs::create_dir_all(&directory).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = directory.join(format!("encoder-{}-{nonce}.mp4", std::process::id()));
    let spec = VideoSpec {
        width: 32,
        height: 32,
        frames: 24,
        fps_num: 24,
        fps_den: 1,
    };
    // Deliberate encoder test colors. Never presented as AI or taxi output.
    mp4::encode(&ffmpeg, &path, spec, |writer| {
        for i in 0..24 {
            writer.write_frame(i, &vec![i as u8 * 10; 32 * 32 * 3])?;
        }
        Ok(())
    })
    .unwrap();
    let original = std::fs::read(&path).unwrap();
    assert!(mp4::encode(&ffmpeg, &path, spec, |_| panic!()).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success());
    let metadata = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "codec_name=h264",
        "width=32",
        "height=32",
        "r_frame_rate=24/1",
        "nb_read_frames=24",
        "duration=1.000000",
    ] {
        assert!(metadata.contains(expected), "{metadata}");
    }
    assert!(
        Command::new(ffmpeg)
            .args(["-v", "error", "-i"])
            .arg(&path)
            .args(["-f", "null", "-"])
            .status()
            .unwrap()
            .success()
    );
}
