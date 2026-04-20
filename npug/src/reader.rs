use crate::error::{Error, Result};
use crate::generated::npug as fb;
use crate::version;

pub struct TensorView {
    pub name: String,
    pub dtype: fb::DType,
    pub dims: Vec<i64>,
    pub symbol_names: Vec<String>,
    pub buffer: u32,
}

pub struct GraphReader<'a> {
    bytes: &'a [u8],
    graph: fb::Graph<'a>,
}

impl<'a> GraphReader<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::TooSmall { got: bytes.len(), need: 8 });
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

    pub fn abi_version(&self) -> u32 { self.graph.abi_version().0 }
    pub fn target(&self) -> fb::TargetId { self.graph.target() }
    pub fn producer(&self) -> &'a str { self.graph.producer().unwrap_or("") }

    pub fn entry_points(&self) -> Vec<&'a str> {
        self.graph
            .entry_points()
            .map(|v| v.iter().map(|e| e.name().unwrap_or("")).collect())
            .unwrap_or_default()
    }

    pub fn tensors(&self) -> Vec<TensorView> {
        self.graph
            .tensors()
            .map(|v| {
                v.iter()
                    .map(|t| TensorView {
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

    pub fn as_bytes(&self) -> &'a [u8] { self.bytes }
}
