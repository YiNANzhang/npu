use crate::generated::npug as fb;
use crate::version;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

pub struct TensorDesc<'a> {
    pub name: &'a str,
    pub dtype: fb::DType,
    pub dims: &'a [i64],
    pub symbol_names: &'a [&'a str],
    pub buffer: Option<u32>,
}

pub struct GraphBuilder<'a> {
    fbb: FlatBufferBuilder<'a>,
    producer: Option<String>,
    target: fb::TargetId,
    pending_tensors: Vec<PendingTensor>,
    pending_buffers: Vec<Vec<u8>>,
}

struct PendingTensor {
    name: String,
    dtype: fb::DType,
    dims: Vec<i64>,
    symbol_names: Vec<String>,
    buffer: u32,
}

impl<'a> GraphBuilder<'a> {
    pub fn new() -> Self {
        Self {
            fbb: FlatBufferBuilder::with_capacity(4096),
            producer: None,
            target: fb::TargetId::AutoSocV1,
            pending_tensors: Vec::new(),
            pending_buffers: Vec::new(),
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
                fb::Tensor::create(
                    &mut self.fbb,
                    &fb::TensorArgs {
                        name: Some(name),
                        dtype: t.dtype,
                        shape: Some(shape),
                        buffer: t.buffer,
                    },
                )
            })
            .collect();
        let tensors_vec = self.fbb.create_vector(&tensor_offsets);

        let producer = self.producer.take().map(|s| self.fbb.create_string(&s));
        let empty_eps: Vec<WIPOffset<fb::EntryPoint>> = Vec::new();
        let entry_points = self.fbb.create_vector(&empty_eps);

        let graph = fb::Graph::create(
            &mut self.fbb,
            &fb::GraphArgs {
                abi_version: fb::AbiVersion(version::CURRENT),
                target: self.target,
                producer,
                tensors: Some(tensors_vec),
                buffers: Some(buffers_vec),
                entry_points: Some(entry_points),
            },
        );

        self.fbb.finish(graph, Some(std::str::from_utf8(version::MAGIC).unwrap()));
        self.fbb.finished_data().to_vec()
    }
}

impl<'a> Default for GraphBuilder<'a> {
    fn default() -> Self { Self::new() }
}
