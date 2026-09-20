use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aster"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn cli_contract() {
    assert!(run(&["help"]).status.success());
    let backend = run(&["backend-smoke"]);
    assert!(backend.status.success());
    assert!(
        String::from_utf8(backend.stdout)
            .unwrap()
            .contains("gradient_max_abs_error=0")
    );
    let device = run(&["device"]);
    assert!(device.status.success());
    assert!(
        String::from_utf8(device.stdout)
            .unwrap()
            .contains("cuda=not_implemented")
    );
    for args in [
        vec!["synthetic", "--frames", "0"],
        vec!["synthetic", "--width"],
        vec!["synthetic", "--bogus", "1"],
        vec!["device", "extra"],
        vec!["synthetic", "--seed", "1", "--seed", "2"],
        vec!["synthetic", "--budget-mib", "0"],
    ] {
        let output = run(&args);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .starts_with("level=error code=")
        );
    }
    let a = run(&["synthetic"]);
    let b = run(&["synthetic"]);
    assert!(a.status.success() && b.status.success());
    let checksum = |o: Vec<u8>| {
        String::from_utf8(o)
            .unwrap()
            .lines()
            .find(|s| s.starts_with("checksum="))
            .unwrap()
            .to_owned()
    };
    assert_eq!(checksum(a.stdout), checksum(b.stdout));
}

#[test]
fn model_inspection_and_media_no_overwrite() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-artifacts")
        .join(format!(
            "cli-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&directory).unwrap();
    let model = directory.join("test.safetensors");
    let header = br#"{"test":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(0_f32.to_le_bytes());
    std::fs::write(&model, &bytes).unwrap();
    let inspected = run(&["inspect", model.to_str().unwrap()]);
    assert!(inspected.status.success());
    let text = String::from_utf8(inspected.stdout).unwrap();
    assert!(text.contains("tensors=1") && text.contains("inference=unsupported"));
    bytes.pop();
    std::fs::write(&model, bytes).unwrap();
    assert!(!run(&["inspect", model.to_str().unwrap()]).status.success());
    let video = directory.join("test.y4m");
    assert!(
        run(&["demo-video", video.to_str().unwrap()])
            .status
            .success()
    );
    let original = std::fs::read(&video).unwrap();
    assert!(original.starts_with(b"YUV4MPEG2 W320 H180 F24:1"));
    assert!(
        !run(&["demo-video", video.to_str().unwrap()])
            .status
            .success()
    );
    assert_eq!(std::fs::read(video).unwrap(), original);
}
