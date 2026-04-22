use npug::builder::GraphBuilder;

#[test]
fn c_abi_version_matches() {
    let v = npug::npug_abi_version();
    assert_eq!(v, npug::version::CURRENT);
}

#[test]
fn c_validate_accepts_good() {
    let bytes = GraphBuilder::new().finish();
    let ok = unsafe { npug::npug_validate(bytes.as_ptr(), bytes.len()) };
    assert_eq!(ok, 1);
}

#[test]
fn c_validate_rejects_short() {
    let bytes = [0u8; 4];
    let ok = unsafe { npug::npug_validate(bytes.as_ptr(), bytes.len()) };
    assert_eq!(ok, 0);
}
