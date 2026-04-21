use npug::builder::GraphBuilder;
use npug::reader::GraphReader;
use npug::Error;

#[test]
fn reject_bad_magic() {
    let mut bytes = GraphBuilder::new().finish();
    bytes[4] = b'X'; // corrupt magic
    match GraphReader::from_bytes(&bytes) {
        Err(Error::BadMagic(m)) => assert_eq!(&m[..1], b"X"),
        Ok(_) => panic!("expected BadMagic, got Ok"),
        Err(e) => panic!("expected BadMagic, got {e}"),
    }
}

#[test]
fn reject_future_major_version() {
    use flatbuffers::FlatBufferBuilder;
    use npug::__generated_for_test as fb;

    let mut fbb = FlatBufferBuilder::with_capacity(256);
    let g = fb::Graph::create(
        &mut fbb,
        &fb::GraphArgs {
            abi_version: fb::AbiVersion(0x010000), // v1.0.0 (future major)
            target: fb::TargetId::AutoSocV1,
            producer: None,
            tensors: None,
            buffers: None,
            kernels: None,
            entry_points: None,
        },
    );
    fbb.finish(g, Some("NPUG"));
    let bytes = fbb.finished_data().to_vec();

    match GraphReader::from_bytes(&bytes) {
        Err(Error::IncompatibleAbi { file, .. }) => assert_eq!(file, 0x010000),
        Ok(_) => panic!("expected IncompatibleAbi, got Ok"),
        Err(e) => panic!("expected IncompatibleAbi, got {e}"),
    }
}

#[test]
fn buffer_too_small() {
    let r = GraphReader::from_bytes(&[0u8; 4]);
    assert!(matches!(r, Err(Error::TooSmall { .. })));
}
