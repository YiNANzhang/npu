//! ABI version — single source of truth.

pub const CURRENT: u32 = 0x00_01_00; // v0.1.0
pub const MAGIC: &[u8; 4] = b"NPUG";

pub fn is_compatible(other: u32) -> bool {
    // Major must match; minor is forward-compatible.
    (CURRENT >> 16) == (other >> 16) && CURRENT >= other
}
