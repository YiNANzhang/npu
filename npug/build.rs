use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let schema = manifest.join("schema/npug.fbs");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", schema.display());

    let status = Command::new("flatc")
        .args([
            "--rust",
            "--gen-mutable",
            "--gen-object-api",
            "-o",
        ])
        .arg(&out_dir)
        .arg(&schema)
        .status()
        .expect("failed to invoke flatc; install flatbuffers");

    assert!(status.success(), "flatc exited with {:?}", status);

    let generated = out_dir.join("npug_generated.rs");
    let dst = manifest.join("src/generated.rs");
    std::fs::copy(&generated, &dst)
        .unwrap_or_else(|e| panic!("copy {} -> {}: {e}", generated.display(), dst.display()));
}
