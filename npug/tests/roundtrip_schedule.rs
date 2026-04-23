use npug::builder::{
    BucketDesc, EntryPointDesc, GraphBuilder, KernelDesc, ScheduleEntryDesc, TensorDesc,
};
use npug::reader::GraphReader;
use npug::{DType, KernelKind};

#[test]
fn roundtrip_single_bucket_entry() {
    let mut b = GraphBuilder::new();
    let in_t = b.add_tensor(TensorDesc {
        name: "in",
        dtype: DType::Bf16,
        dims: &[1, 128],
        symbol_names: &[],
        buffer: None,
    });
    let out_t = b.add_tensor(TensorDesc {
        name: "out",
        dtype: DType::Bf16,
        dims: &[1, 128],
        symbol_names: &[],
        buffer: None,
    });
    let buf = b.add_buffer(&[0u8; 16]);
    let kern = b.add_kernel(KernelDesc {
        name: "passthrough",
        kind: KernelKind::TileUcBin,
        buffer: buf,
        entry_offset: 0,
    });

    let ep = b.add_entry_point(EntryPointDesc {
        name: "perception",
        inputs: &[in_t],
        outputs: &[out_t],
        buckets: &[BucketDesc {
            shape_hint_dims: &[1, 128],
            schedule: &[ScheduleEntryDesc {
                tile_id: 0,
                kernel_index: kern,
                args_offset: 0,
                args_size: 0,
            }],
        }],
    });
    assert_eq!(ep, 0);

    let bytes = b.finish();
    let r = GraphReader::from_bytes(&bytes).unwrap();
    let eps = r.entry_points_full();
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].name, "perception");
    assert_eq!(eps[0].inputs, vec![in_t]);
    assert_eq!(eps[0].outputs, vec![out_t]);
    assert_eq!(eps[0].buckets.len(), 1);
    assert_eq!(eps[0].buckets[0].shape_hint_dims, vec![1, 128]);
    assert_eq!(eps[0].buckets[0].schedule.len(), 1);
    assert_eq!(eps[0].buckets[0].schedule[0].tile_id, 0);
    assert_eq!(eps[0].buckets[0].schedule[0].kernel_index, kern);
}
