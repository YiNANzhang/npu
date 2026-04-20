use npug::builder::{GraphBuilder, QuantDesc, TensorDesc};
use npug::reader::GraphReader;
use npug::{DType, MemoryRegion, QuantScheme};

#[test]
fn roundtrip_mx_block_quant() {
    let mut b = GraphBuilder::new();
    let weight_buf = b.add_buffer(&vec![0u8; 1024]);
    let scale_buf = b.add_buffer(&vec![0x40u8; 32]); // 32 E8M0 scales

    let t = b.add_tensor(TensorDesc {
        name: "w_qkv",
        dtype: DType::MxFp4,
        dims: &[4096, 4096],
        symbol_names: &[],
        buffer: Some(weight_buf),
    });
    b.attach_quant(t, QuantDesc {
        scheme: QuantScheme::MxBlock32,
        scale_buffer: Some(scale_buf),
        zero_point_buffer: None,
        axis: -1,
        block_size: 32,
    });
    b.set_region(t, MemoryRegion::Lpddr, 0x1000_0000);

    let bytes = b.finish();
    let r = GraphReader::from_bytes(&bytes).unwrap();
    let t = &r.tensors()[0];
    assert_eq!(t.quant.scheme, QuantScheme::MxBlock32);
    assert_eq!(t.quant.scale_buffer, Some(scale_buf));
    assert_eq!(t.quant.block_size, 32);
    assert_eq!(t.region, MemoryRegion::Lpddr);
    assert_eq!(t.offset, 0x1000_0000);
}

#[test]
fn roundtrip_no_quant() {
    let mut b = GraphBuilder::new();
    b.add_tensor(TensorDesc {
        name: "act",
        dtype: DType::Bf16,
        dims: &[1, 128],
        symbol_names: &[],
        buffer: None,
    });
    let bytes = b.finish();
    let r = GraphReader::from_bytes(&bytes).unwrap();
    assert_eq!(r.tensors()[0].quant.scheme, QuantScheme::None);
}
