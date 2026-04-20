use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid npug magic; expected NPUG, got {0:?}")]
    BadMagic([u8; 4]),

    #[error("incompatible ABI version: file=0x{file:06x}, crate=0x{crate_version:06x}")]
    IncompatibleAbi { file: u32, crate_version: u32 },

    #[error("flatbuffers parse error: {0}")]
    Parse(#[from] flatbuffers::InvalidFlatbuffer),

    #[error("buffer too small: {got} bytes, need ≥ {need}")]
    TooSmall { got: usize, need: usize },
}

pub type Result<T> = std::result::Result<T, Error>;
