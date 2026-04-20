use npug::{builder::GraphBuilder, reader::GraphReader, version};

#[test]
fn roundtrip_empty_graph() {
    let mut b = GraphBuilder::new();
    b.set_producer("npuc-test/0.1");
    let bytes = b.finish();

    assert!(bytes.len() >= 32, "npug too small: {}", bytes.len());
    assert_eq!(&bytes[4..8], b"NPUG", "missing file identifier");

    let r = GraphReader::from_bytes(&bytes).expect("parse");
    assert_eq!(r.abi_version(), version::CURRENT);
    assert_eq!(r.producer(), "npuc-test/0.1");
    assert_eq!(r.entry_points().len(), 0);
}
