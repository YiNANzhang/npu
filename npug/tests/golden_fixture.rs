use npug::reader::GraphReader;
use std::io::Read;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/minimal_v0_1.npug");

// SHA-256 of `fixtures/minimal_v0_1.npug`. If this drifts, either (a) you accidentally
// changed the wire format — revert, or (b) it's intentional — regen fixture, bump ABI,
// update this constant.
const GOLDEN_SHA256: &str = "8f591b7530f78e4919bedb31a7882093eb1b6e0ef6216ceaf35f403c5cdc4ebc";

fn sha256_hex(bytes: &[u8]) -> String {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut c = Command::new("shasum").args(["-a", "256"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    c.stdin.as_mut().unwrap().write_all(bytes).unwrap();
    let out = c.wait_with_output().unwrap();
    let s = String::from_utf8(out.stdout).unwrap();
    s.split_whitespace().next().unwrap().to_string()
}

#[test]
fn golden_fixture_hash_stable() {
    let mut f = std::fs::File::open(FIXTURE).expect("fixture missing; run `cargo run --example gen_golden`");
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).unwrap();
    let got = sha256_hex(&bytes);
    assert_eq!(got, GOLDEN_SHA256,
        "golden .npug hash drifted — if this is intentional, regen fixture and update GOLDEN_SHA256 (and bump ABI)");
}

#[test]
fn golden_fixture_parses() {
    let bytes = std::fs::read(FIXTURE).unwrap();
    let r = GraphReader::from_bytes(&bytes).unwrap();
    assert_eq!(r.abi_version(), npug::version::CURRENT);
    assert_eq!(r.producer(), "npug-golden-gen/0.1.0");
    assert_eq!(r.tensors().len(), 3);
    assert_eq!(r.kernels().len(), 1);
    assert_eq!(r.entry_points_full().len(), 1);
}
