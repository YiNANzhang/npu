use npug::builder::{GraphBuilder, KernelDesc};
use npug::reader::GraphReader;
use npug::KernelKind;

#[test]
fn roundtrip_tile_uc_kernel() {
    let mut b = GraphBuilder::new();
    let code = vec![0x93, 0x00, 0x00, 0x00]; // RV32 nop
    let buf = b.add_buffer(&code);
    let k = b.add_kernel(KernelDesc {
        name: "gemm_mxfp4_128x128",
        kind: KernelKind::TileUcBin,
        buffer: buf,
        entry_offset: 0,
    });
    assert_eq!(k, 0);
    let bytes = b.finish();

    let r = GraphReader::from_bytes(&bytes).unwrap();
    let kernels = r.kernels();
    assert_eq!(kernels.len(), 1);
    assert_eq!(kernels[0].name, "gemm_mxfp4_128x128");
    assert_eq!(kernels[0].kind, KernelKind::TileUcBin);
    assert_eq!(kernels[0].entry_offset, 0);
}
