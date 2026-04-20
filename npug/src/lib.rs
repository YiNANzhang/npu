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
