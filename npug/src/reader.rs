use crate::error::{Error, Result};
use crate::generated::npug as fb;
use crate::version;

pub struct ScheduleEntryView {
    pub tile_id: u32,
    pub kernel_index: u32,
    pub args_offset: u64,
    pub args_size: u32,
}

pub struct BucketView {
    pub shape_hint_dims: Vec<i64>,
    pub schedule: Vec<ScheduleEntryView>,
}

pub struct EntryPointView {
    pub name: String,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub buckets: Vec<BucketView>,
}

pub struct KernelView {
    pub name: String,
    pub kind: fb::KernelKind,
    pub buffer: u32,
    pub entry_offset: u64,
}

pub struct QuantView {
    pub scheme: fb::QuantScheme,
    pub scale_buffer: Option<u32>,
    pub zero_point_buffer: Option<u32>,
    pub axis: i32,
    pub block_size: u32,
}

pub struct TensorView {
    pub name: String,
    pub dtype: fb::DType,
    pub dims: Vec<i64>,
    pub symbol_names: Vec<String>,
    pub buffer: u32,
    pub quant: QuantView,
    pub region: fb::MemoryRegion,
    pub offset: u64,
}

pub struct GraphReader<'a> {
    bytes: &'a [u8],
    graph: fb::Graph<'a>,
}

impl<'a> GraphReader<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::TooSmall {
                got: bytes.len(),
                need: 8,
            });
        }
        let magic: [u8; 4] = bytes[4..8].try_into().unwrap();
        if &magic != version::MAGIC {
            return Err(Error::BadMagic(magic));
        }
        let graph = flatbuffers::root::<fb::Graph>(bytes)?;
        let abi = graph.abi_version().0;
        if !version::is_compatible(abi) {
            return Err(Error::IncompatibleAbi {
                file: abi,
                crate_version: version::CURRENT,
            });
        }
        Ok(Self { bytes, graph })
    }

    pub fn abi_version(&self) -> u32 {
        self.graph.abi_version().0
    }
    pub fn target(&self) -> fb::TargetId {
        self.graph.target()
    }
    pub fn producer(&self) -> &'a str {
        self.graph.producer().unwrap_or("")
    }

    pub fn entry_points(&self) -> Vec<&'a str> {
        self.graph
            .entry_points()
            .map(|v| v.iter().map(|e| e.name().unwrap_or("")).collect())
            .unwrap_or_default()
    }

    pub fn entry_points_full(&self) -> Vec<EntryPointView> {
        self.graph
            .entry_points()
            .map(|v| {
                v.iter()
                    .map(|ep| {
                        let buckets = ep
                            .buckets()
                            .map(|bv| {
                                bv.iter()
                                    .map(|b| {
                                        let shape_hint_dims = b
                                            .shape_hint()
                                            .and_then(|s| s.dims())
                                            .map(|d| d.iter().collect())
                                            .unwrap_or_default();
                                        let schedule = b
                                            .schedule()
                                            .map(|sv| {
                                                sv.iter()
                                                    .map(|e| ScheduleEntryView {
                                                        tile_id: e.tile_id(),
                                                        kernel_index: e.kernel_index(),
                                                        args_offset: e.args_offset(),
                                                        args_size: e.args_size(),
                                                    })
                                                    .collect()
                                            })
                                            .unwrap_or_default();
                                        BucketView {
                                            shape_hint_dims,
                                            schedule,
                                        }
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        EntryPointView {
                            name: ep.name().unwrap_or("").to_string(),
                            inputs: ep.inputs().map(|v| v.iter().collect()).unwrap_or_default(),
                            outputs: ep.outputs().map(|v| v.iter().collect()).unwrap_or_default(),
                            buckets,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn tensors(&self) -> Vec<TensorView> {
        self.graph
            .tensors()
            .map(|v| {
                v.iter()
                    .map(|t| {
                        let q = t.quant();
                        TensorView {
                            name: t.name().unwrap_or("").to_string(),
                            dtype: t.dtype(),
                            dims: t
                                .shape()
                                .and_then(|s| s.dims())
                                .map(|d| d.iter().collect())
                                .unwrap_or_default(),
                            symbol_names: t
                                .shape()
                                .and_then(|s| s.symbol_names())
                                .map(|v| v.iter().map(|s| s.to_string()).collect())
                                .unwrap_or_default(),
                            buffer: t.buffer(),
                            quant: QuantView {
                                scheme: q.map(|q| q.scheme()).unwrap_or(fb::QuantScheme::None),
                                scale_buffer: q.and_then(|q| {
                                    (q.scale_buffer() != u32::MAX).then_some(q.scale_buffer())
                                }),
                                zero_point_buffer: q.and_then(|q| {
                                    (q.zero_point_buffer() != u32::MAX)
                                        .then_some(q.zero_point_buffer())
                                }),
                                axis: q.map(|q| q.axis()).unwrap_or(-1),
                                block_size: q.map(|q| q.block_size()).unwrap_or(0),
                            },
                            region: t.region(),
                            offset: t.offset(),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn buffer_count(&self) -> usize {
        self.graph.buffers().map(|v| v.len()).unwrap_or(0)
    }

    pub fn buffer_bytes(&self, idx: u32) -> &'a [u8] {
        self.graph
            .buffers()
            .and_then(|v| v.get(idx as usize).data())
            .map(|d| d.bytes())
            .unwrap_or(&[])
    }

    pub fn kernels(&self) -> Vec<KernelView> {
        self.graph
            .kernels()
            .map(|v| {
                v.iter()
                    .map(|k| KernelView {
                        name: k.name().unwrap_or("").to_string(),
                        kind: k.kind(),
                        buffer: k.buffer(),
                        entry_offset: k.entry_offset(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }
}
