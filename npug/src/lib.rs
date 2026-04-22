//! .npug — Auto-NPU deployment container (FlatBuffer)
//!
//! ABI version: v0.1 (see `docs/npug-abi-v0.1.md`)

#![allow(clippy::all, unused)]

extern crate alloc;

mod generated;
pub mod version;

pub mod error;
pub mod builder;
pub mod reader;

pub use error::{Error, Result};
pub use generated::npug::DType;
pub use generated::npug::TargetId;
pub use generated::npug::{MemoryRegion, QuantScheme};
pub use generated::npug::KernelKind;

#[doc(hidden)]
pub mod __generated_for_test {
    pub use super::generated::npug::*;
}

/// Opaque handle to a parsed .npug.
#[repr(C)]
pub struct NpugHandle(*const u8);

/// Returns the crate's ABI version (e.g. 0x000100).
#[no_mangle]
pub extern "C" fn npug_abi_version() -> u32 {
    version::CURRENT
}

/// Returns 1 if `bytes[..len]` parses as a valid .npug with compatible ABI, else 0.
///
/// # Safety
/// `bytes` must be valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn npug_validate(bytes: *const u8, len: usize) -> i32 {
    if bytes.is_null() { return 0; }
    let slice = std::slice::from_raw_parts(bytes, len);
    match reader::GraphReader::from_bytes(slice) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}
