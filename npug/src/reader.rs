use crate::error::{Error, Result};
use crate::generated::npug as fb;
use crate::version;

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
    pub fn as_bytes(&self) -> &'a [u8] { self.bytes }
}
