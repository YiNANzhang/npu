use crate::generated::npug as fb;
use crate::version;
use flatbuffers::FlatBufferBuilder;

pub struct TensorDesc<'a> {
    pub name: &'a str,
    pub dtype: fb::DType,
    pub dims: &'a [i64],
    pub symbol_names: &'a [&'a str],
    pub buffer: Option<u32>,
}

pub struct QuantDesc {
    pub scheme: fb::QuantScheme,
    pub scale_buffer: Option<u32>,
    pub zero_point_buffer: Option<u32>,
    pub axis: i32,
    pub block_size: u32,
}

impl Default for QuantDesc {
    fn default() -> Self {
        Self {
            scheme: fb::QuantScheme::None,
            scale_buffer: None,
            zero_point_buffer: None,
            axis: -1,
            block_size: 0,
        }
    }
}

pub struct KernelDesc<'a> {
    pub name: &'a str,
    pub kind: fb::KernelKind,
    pub buffer: u32,
    pub entry_offset: u64,
}

struct PendingKernel {
    name: String,
    kind: fb::KernelKind,
    buffer: u32,
    entry_offset: u64,
}

pub struct ScheduleEntryDesc {
    pub tile_id: u32,
    pub kernel_index: u32,
    pub args_offset: u64,
    pub args_size: u32,
}

pub struct BucketDesc<'a> {
    pub shape_hint_dims: &'a [i64],
    pub schedule: &'a [ScheduleEntryDesc],
}

pub struct EntryPointDesc<'a> {
    pub name: &'a str,
    pub inputs: &'a [u32],
    pub outputs: &'a [u32],
    pub buckets: &'a [BucketDesc<'a>],
}

struct PendingEntryPoint {
    name: String,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    buckets: Vec<PendingBucket>,
}

struct PendingBucket {
    shape_hint_dims: Vec<i64>,
    schedule: Vec<ScheduleEntryDesc>,
}

pub struct GraphBuilder<'a> {
    fbb: FlatBufferBuilder<'a>,
    producer: Option<String>,
    target: fb::TargetId,
    pending_tensors: Vec<PendingTensor>,
    pending_buffers: Vec<Vec<u8>>,
    pending_kernels: Vec<PendingKernel>,
    pending_entry_points: Vec<PendingEntryPoint>,
}

struct PendingTensor {
    name: String,
    dtype: fb::DType,
    dims: Vec<i64>,
    symbol_names: Vec<String>,
    buffer: u32,
    quant: QuantDesc,
    region: fb::MemoryRegion,
    offset: u64,
}

impl<'a> GraphBuilder<'a> {
    pub fn new() -> Self {
        Self {
            fbb: FlatBufferBuilder::with_capacity(4096),
            producer: None,
            target: fb::TargetId::AutoSocV1,
            pending_tensors: Vec::new(),
            pending_buffers: Vec::new(),
            pending_kernels: Vec::new(),
            pending_entry_points: Vec::new(),
        }
    }

    pub fn set_producer(&mut self, s: &str) -> &mut Self {
        self.producer = Some(s.to_string());
        self
    }

    pub fn set_target(&mut self, t: fb::TargetId) -> &mut Self {
        self.target = t;
        self
    }

    pub fn add_buffer(&mut self, data: &[u8]) -> u32 {
        let idx = self.pending_buffers.len() as u32;
        self.pending_buffers.push(data.to_vec());
        idx
    }

    pub fn add_kernel(&mut self, d: KernelDesc<'_>) -> u32 {
        let idx = self.pending_kernels.len() as u32;
        self.pending_kernels.push(PendingKernel {
            name: d.name.to_string(),
            kind: d.kind,
            buffer: d.buffer,
            entry_offset: d.entry_offset,
        });
        idx
    }

    pub fn add_tensor(&mut self, desc: TensorDesc<'_>) -> u32 {
        assert!(
            desc.symbol_names.is_empty() || desc.symbol_names.len() == desc.dims.len(),
            "symbol_names length must be 0 or equal to dims"
        );
        let idx = self.pending_tensors.len() as u32;
        self.pending_tensors.push(PendingTensor {
            name: desc.name.to_string(),
            dtype: desc.dtype,
            dims: desc.dims.to_vec(),
            symbol_names: desc.symbol_names.iter().map(|s| s.to_string()).collect(),
            buffer: desc.buffer.unwrap_or(u32::MAX),
            quant: QuantDesc::default(),
            region: fb::MemoryRegion::Unknown,
            offset: 0,
        });
        idx
    }

    pub fn attach_quant(&mut self, tensor_idx: u32, q: QuantDesc) -> &mut Self {
        self.pending_tensors[tensor_idx as usize].quant = q;
        self
    }

    pub fn set_region(
        &mut self,
        tensor_idx: u32,
        region: fb::MemoryRegion,
        offset: u64,
    ) -> &mut Self {
        self.pending_tensors[tensor_idx as usize].region = region;
        self.pending_tensors[tensor_idx as usize].offset = offset;
        self
    }

    pub fn add_entry_point(&mut self, d: EntryPointDesc<'_>) -> u32 {
        let idx = self.pending_entry_points.len() as u32;
        self.pending_entry_points.push(PendingEntryPoint {
            name: d.name.to_string(),
            inputs: d.inputs.to_vec(),
            outputs: d.outputs.to_vec(),
            buckets: d
                .buckets
                .iter()
                .map(|b| PendingBucket {
                    shape_hint_dims: b.shape_hint_dims.to_vec(),
                    schedule: b
                        .schedule
                        .iter()
                        .map(|e| ScheduleEntryDesc {
                            tile_id: e.tile_id,
                            kernel_index: e.kernel_index,
                            args_offset: e.args_offset,
                            args_size: e.args_size,
                        })
                        .collect(),
                })
                .collect(),
        });
        idx
    }

    pub fn finish(mut self) -> Vec<u8> {
        // Buffers
        let buf_offsets: Vec<_> = self
            .pending_buffers
            .iter()
            .map(|data| {
                let d = self.fbb.create_vector::<u8>(data);
                fb::Buffer::create(&mut self.fbb, &fb::BufferArgs { data: Some(d) })
            })
            .collect();
        let buffers_vec = self.fbb.create_vector(&buf_offsets);

        // Tensors
        let tensor_offsets: Vec<_> = self
            .pending_tensors
            .iter()
            .map(|t| {
                let name = self.fbb.create_string(&t.name);
                let dims = self.fbb.create_vector(&t.dims);
                let symbol_offsets: Vec<_> = t
                    .symbol_names
                    .iter()
                    .map(|s| self.fbb.create_string(s))
                    .collect();
                let symbol_vec = self.fbb.create_vector(&symbol_offsets);
                let shape = fb::Shape::create(
                    &mut self.fbb,
                    &fb::ShapeArgs {
                        dims: Some(dims),
                        symbol_names: Some(symbol_vec),
                    },
                );
                let quant = fb::QuantInfo::create(
                    &mut self.fbb,
                    &fb::QuantInfoArgs {
                        scheme: t.quant.scheme,
                        scale_buffer: t.quant.scale_buffer.unwrap_or(u32::MAX),
                        zero_point_buffer: t.quant.zero_point_buffer.unwrap_or(u32::MAX),
                        axis: t.quant.axis,
                        block_size: t.quant.block_size,
                    },
                );
                fb::Tensor::create(
                    &mut self.fbb,
                    &fb::TensorArgs {
                        name: Some(name),
                        dtype: t.dtype,
                        shape: Some(shape),
                        buffer: t.buffer,
                        quant: Some(quant),
                        region: t.region,
                        offset: t.offset,
                    },
                )
            })
            .collect();
        let tensors_vec = self.fbb.create_vector(&tensor_offsets);

        let kernel_offsets: Vec<_> = self
            .pending_kernels
            .iter()
            .map(|k| {
                let name = self.fbb.create_string(&k.name);
                fb::Kernel::create(
                    &mut self.fbb,
                    &fb::KernelArgs {
                        name: Some(name),
                        kind: k.kind,
                        buffer: k.buffer,
                        entry_offset: k.entry_offset,
                    },
                )
            })
            .collect();
        let kernels_vec = self.fbb.create_vector(&kernel_offsets);

        let producer = self.producer.take().map(|s| self.fbb.create_string(&s));

        // Entry points
        let ep_offsets: Vec<_> = self
            .pending_entry_points
            .iter()
            .map(|ep| {
                let name = self.fbb.create_string(&ep.name);
                let inputs = self.fbb.create_vector(&ep.inputs);
                let outputs = self.fbb.create_vector(&ep.outputs);

                let bucket_offsets: Vec<_> = ep
                    .buckets
                    .iter()
                    .map(|b| {
                        let dims = self.fbb.create_vector(&b.shape_hint_dims);
                        let empty_syms: Vec<flatbuffers::WIPOffset<&str>> = Vec::new();
                        let symbol_names = self.fbb.create_vector(&empty_syms);
                        let shape = fb::Shape::create(
                            &mut self.fbb,
                            &fb::ShapeArgs {
                                dims: Some(dims),
                                symbol_names: Some(symbol_names),
                            },
                        );

                        let sched_offsets: Vec<_> = b
                            .schedule
                            .iter()
                            .map(|e| {
                                fb::ScheduleEntry::create(
                                    &mut self.fbb,
                                    &fb::ScheduleEntryArgs {
                                        tile_id: e.tile_id,
                                        kernel_index: e.kernel_index,
                                        args_offset: e.args_offset,
                                        args_size: e.args_size,
                                    },
                                )
                            })
                            .collect();
                        let schedule = self.fbb.create_vector(&sched_offsets);

                        fb::Bucket::create(
                            &mut self.fbb,
                            &fb::BucketArgs {
                                shape_hint: Some(shape),
                                schedule: Some(schedule),
                            },
                        )
                    })
                    .collect();
                let buckets = self.fbb.create_vector(&bucket_offsets);

                fb::EntryPoint::create(
                    &mut self.fbb,
                    &fb::EntryPointArgs {
                        name: Some(name),
                        inputs: Some(inputs),
                        outputs: Some(outputs),
                        buckets: Some(buckets),
                    },
                )
            })
            .collect();
        let entry_points = self.fbb.create_vector(&ep_offsets);

        let graph = fb::Graph::create(
            &mut self.fbb,
            &fb::GraphArgs {
                abi_version: fb::AbiVersion(version::CURRENT),
                target: self.target,
                producer,
                tensors: Some(tensors_vec),
                buffers: Some(buffers_vec),
                kernels: Some(kernels_vec),
                entry_points: Some(entry_points),
            },
        );

        self.fbb
            .finish(graph, Some(std::str::from_utf8(version::MAGIC).unwrap()));
        self.fbb.finished_data().to_vec()
    }
}

impl<'a> Default for GraphBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}
