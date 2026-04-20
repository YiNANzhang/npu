use npug::builder::{GraphBuilder, TensorDesc};
use npug::reader::GraphReader;

#[test]
fn roundtrip_static_tensor() {
    let mut b = GraphBuilder::new();
    let buf_idx = b.add_buffer(&[1u8, 2, 3, 4, 5, 6, 7, 8]);
    let t_idx = b.add_tensor(TensorDesc {
        name: "input",
        dtype: npug::DType::Bf16,
        dims: &[1, 4],
        symbol_names: &[],
        buffer: Some(buf_idx),
    });
    assert_eq!(t_idx, 0);

    let bytes = b.finish();

    let r = GraphReader::from_bytes(&bytes).unwrap();
    let tensors = r.tensors();
    assert_eq!(tensors.len(), 1);
    let t = &tensors[0];
    assert_eq!(t.name, "input");
    assert_eq!(t.dtype, npug::DType::Bf16);
    assert_eq!(t.dims, vec![1, 4]);
    assert_eq!(r.buffer_bytes(buf_idx), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn roundtrip_symbolic_shape() {
    let mut b = GraphBuilder::new();
    b.add_tensor(TensorDesc {
        name: "seq",
        dtype: npug::DType::Fp16,
        dims: &[1, -1, 4096],
        symbol_names: &["", "S", ""],
        buffer: None,
    });
    let bytes = b.finish();
    let r = GraphReader::from_bytes(&bytes).unwrap();
    let t = &r.tensors()[0];
    assert_eq!(t.dims, vec![1, -1, 4096]);
    assert_eq!(t.symbol_names, vec!["".to_string(), "S".to_string(), "".to_string()]);
}
