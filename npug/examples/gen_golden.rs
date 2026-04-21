use npug::builder::{BucketDesc, EntryPointDesc, GraphBuilder, KernelDesc, QuantDesc,
                    ScheduleEntryDesc, TensorDesc};
use npug::{DType, KernelKind, MemoryRegion, QuantScheme};

fn main() {
    let mut b = GraphBuilder::new();
    b.set_producer("npug-golden-gen/0.1.0");

    let w_buf = b.add_buffer(&vec![0u8; 64]);
    let s_buf = b.add_buffer(&vec![0x40u8; 2]);
    let in_t = b.add_tensor(TensorDesc {
        name: "x", dtype: DType::Bf16, dims: &[1, 16], symbol_names: &[], buffer: None,
    });
    let w_t = b.add_tensor(TensorDesc {
        name: "w", dtype: DType::MxFp4, dims: &[16, 16], symbol_names: &[],
        buffer: Some(w_buf),
    });
    b.attach_quant(w_t, QuantDesc {
        scheme: QuantScheme::MxBlock32,
        scale_buffer: Some(s_buf),
        zero_point_buffer: None,
        axis: -1,
        block_size: 32,
    });
    b.set_region(w_t, MemoryRegion::Lpddr, 0x0);
    let out_t = b.add_tensor(TensorDesc {
        name: "y", dtype: DType::Bf16, dims: &[1, 16], symbol_names: &[], buffer: None,
    });

    let kbuf = b.add_buffer(&[0x13, 0x00, 0x00, 0x00]); // RV32 nop
    let kern = b.add_kernel(KernelDesc {
        name: "gemm_nop", kind: KernelKind::TileUcBin, buffer: kbuf, entry_offset: 0,
    });

    b.add_entry_point(EntryPointDesc {
        name: "main",
        inputs: &[in_t],
        outputs: &[out_t],
        buckets: &[BucketDesc {
            shape_hint_dims: &[1, 16],
            schedule: &[ScheduleEntryDesc {
                tile_id: 0, kernel_index: kern, args_offset: 0, args_size: 0,
            }],
        }],
    });

    let bytes = b.finish();
    let path = "npug/fixtures/minimal_v0_1.npug";
    std::fs::write(path, &bytes).unwrap();
    eprintln!("wrote {} ({} bytes)", path, bytes.len());
}
