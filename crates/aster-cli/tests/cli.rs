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
        vec!["control"],
        vec![
            "control",
            "--worker",
            "missing-worker",
            "--models",
            "missing-models",
            "--run-dir",
            "unused",
            "--ffmpeg",
            "missing",
            "--stage",
            "14",
        ],
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
fn text_planning_and_no_fake_generation() {
    let prompt = aster_core::planner::TAXI_PROMPT;
    let output = run(&["plan", "--prompt", prompt]);
    assert!(output.status.success(), "{:?}", output);
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("action=enter") && text.contains("action=depart"));
    assert_eq!(text.matches("generation_mode=Full").count(), 6);
    assert!(text.contains("neural_inference=false"));
    assert!(!run(&["plan", "--prompt", "A dog swims"]).status.success());
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-artifacts");
    std::fs::create_dir_all(&directory).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let source = directory.join(format!("prompt-{}-{nonce}.txt", std::process::id()));
    std::fs::write(&source, prompt).unwrap();
    let from_file = run(&["plan", "--prompt-file", source.to_str().unwrap()]);
    assert!(from_file.status.success());
    assert_eq!(String::from_utf8(from_file.stdout).unwrap(), text);
    let destination = source.with_extension("mp4");
    let generation = run(&[
        "generate",
        "--prompt",
        "My own original scene",
        "--output",
        destination.to_str().unwrap(),
    ]);
    assert!(!generation.status.success());
    assert!(generation.stdout.is_empty());
    assert!(
        String::from_utf8(generation.stderr)
            .unwrap()
            .contains("code=backend_unavailable")
    );
    assert!(!destination.exists());
    for extra in [
        vec!["--reuse", "maybe"],
        vec!["--planner", "taxi-v1"],
        vec!["--reuse", "on", "--reuse", "off"],
    ] {
        let mut args = vec![
            "sequence",
            "--prompt",
            aster_core::planner::CONTROL_PROMPT,
            "--output",
            destination.to_str().unwrap(),
        ];
        args.extend(extra);
        let failed = run(&args);
        assert!(!failed.status.success());
        assert!(
            String::from_utf8(failed.stderr)
                .unwrap()
                .contains("code=invalid_input")
        );
        assert!(!destination.exists());
    }
    std::fs::write(&source, vec![b'x'; 16385]).unwrap();
    assert!(
        !run(&["plan", "--prompt-file", source.to_str().unwrap()])
            .status
            .success()
    );
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
