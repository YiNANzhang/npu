# .npug ABI v0.1 Specification

**Status:** FROZEN as of 2026-04-18. Any wire-format-incompatible change requires major version bump (v1.x).
**Owners:** Compiler team (`npuc`) + Runtime team (`npu-rt`), jointly maintain `schema/npug.fbs`.
**Version encoding:** `(major<<16) | (minor<<8) | patch`. v0.1.0 = `0x000100`.

## Purpose

`.npug` is the binary deployment container emitted by the auto-NPU compiler and consumed by the runtime. It carries:

1. Graph metadata (tensors, entry points, schedule)
2. Weight buffers (including block-scale factors)
3. Kernel binaries (Tile μC + Scheduler MCU RISC-V code)
4. Memory plan (pre-assigned LSRAM / CSRAM / SLC / LPDDR regions)

The format is **FlatBuffers** — zero-copy read, forward-compatible within a major version, language-agnostic.

## File layout

```text
Offset 0x0000 : [4 byte] FlatBuffers root offset
Offset 0x0004 : [4 byte] File identifier "NPUG" (ASCII)
Offset 0x0008 : Root Graph table (FlatBuffer)
Trailing     : Buffer data (weights, kernels, scales)
```

- **File extension:** `.npug`
- **Magic:** `N P U G` at offset 4 (FlatBuffers convention)
- **Endianness:** little-endian (FlatBuffers default)
- **Alignment:** 8-byte for top-level buffer payload

## Compatibility rules

| Change | Allowed within major | Requires major bump |
|---|---|---|
| Add field to existing table (with default) | ✅ | |
| Add new table / enum variant | ✅ | |
| Add new entry to DType / MemoryRegion | ✅ | |
| Remove / rename field | | ✅ |
| Change field type | | ✅ |
| Change semantics of existing field | | ✅ |
| Reorder fields (changes offsets) | | ✅ |

A runtime with ABI crate v0.1.K accepts files produced by ABI v0.1.0 … v0.1.K (forward-compat). Files from v0.2.x are rejected with `IncompatibleAbi`.

## Root: Graph

| Field | Type | Meaning |
|---|---|---|
| `abi_version` | `AbiVersion` (u32) | Must satisfy `(file_major == crate_major) && (crate_version >= file_version)`. |
| `target` | `TargetId` | Must match runtime's target. Unknown → runtime rejects. |
| `producer` | `string` | Free-form compiler version tag. Logged, not enforced. |
| `tensors` | `[Tensor]` | All tensors referenced in the graph. Indexed by entry-point `inputs/outputs`. |
| `buffers` | `[Buffer]` | Raw byte blobs. Referenced by `Tensor.buffer` and `Kernel.buffer`. |
| `kernels` | `[Kernel]` | Tile μC / Sched MCU binary code. Referenced by `ScheduleEntry.kernel_index`. |
| `entry_points` | `[EntryPoint]` | Callable entry points (e.g., `"perception"`, `"prefill"`, `"decode"`). |

## Tensor

| Field | Type | Meaning |
|---|---|---|
| `name` | `string` | Human-readable, debugging only. |
| `dtype` | `DType` | See table below. |
| `shape` | `Shape` | `dims[i]=-1` → symbolic; `symbol_names[i]` gives the symbol. |
| `buffer` | `u32` | Index into `Graph.buffers`; `0xFFFFFFFF` = activation placeholder (no data). |
| `quant` | `QuantInfo` | Quantization metadata. `scheme=None` for non-quantized. |
| `region` | `MemoryRegion` | Assigned memory region. `Unknown` for runtime-planned tensors. |
| `offset` | `u64` | Byte offset within `region`. Ignored if `region=Unknown`. |

### DType

| Code | Name | Encoding |
|---|---|---|
| 1 | `Int4` | 4-bit signed, two-per-byte LE |
| 2 | `Int8` | 8-bit signed |
| 3 | `MxInt4` | OCP MX block=32 + E8M0 scale |
| 4 | `MxInt8` | OCP MX block=32 + E8M0 scale |
| 5 | `MxFp4` | OCP MXFP4 (E2M1), block=32 + E8M0 |
| 6 | `MxFp8` | OCP MXFP8 (E4M3/E5M2 variant pair), block=32 + E8M0 |
| 7 | `MxFp16` | OCP MXFP16 block=32 + E8M0 |
| 8 | `NvFp4` | Nvidia NVFP4, block=16 + E4M3 |
| 9 | `NvFp8` | Nvidia NVFP8, block=16 + E4M3 |
| 10 | `Fp16` | IEEE 754 binary16 |
| 11 | `Bf16` | bfloat16 |
| 12 | `Fp32` | IEEE 754 binary32 |
| 13 | `E8M0Scale` | MX scale sidecar tensor |
| 14 | `E4M3Scale` | NV scale sidecar tensor |

### QuantInfo

| Field | Type | Meaning |
|---|---|---|
| `scheme` | `QuantScheme` | `None / PerTensor / PerChannel / MxBlock32 / NvBlock16` |
| `scale_buffer` | `u32` | Index into `buffers`; holds scales. `0xFFFFFFFF` = none. |
| `zero_point_buffer` | `u32` | Optional zero points (INT schemes only). |
| `axis` | `i32` | Per-channel axis. `-1` for per-tensor or block schemes. |
| `block_size` | `u32` | `32` for MX, `16` for NV, `0` for non-block. |

### MemoryRegion

| Code | Name | Sized | Notes |
|---|---|---|---|
| 1 | `Lsram` | 1 MB per Tile × 64 | Offset is per-Tile; runtime picks Tile via schedule. |
| 2 | `Csram` | 4 MB per Cluster × 8 | Offset is per-Cluster. |
| 3 | `Slc` | 16 MB | Shared SoC-wide. |
| 4 | `Lpddr` | ≤ 32 GB | Physical-address offset. |

## Kernel

| Field | Type | Meaning |
|---|---|---|
| `name` | `string` | Debug only. |
| `kind` | `KernelKind` | `TileUcBin` (RV32E Tile μC) / `SchedMcuBin` (RV32IMACH Scheduler MCU). |
| `buffer` | `u32` | Index into `buffers`; holds the RISC-V image. |
| `entry_offset` | `u64` | Byte offset of entry PC within the buffer. |

## EntryPoint

| Field | Type | Meaning |
|---|---|---|
| `name` | `string` | Runtime dispatch key (e.g., `"perception"`). |
| `inputs` | `[u32]` | Tensor indices. |
| `outputs` | `[u32]` | Tensor indices. |
| `buckets` | `[Bucket]` | ≥ 1. Static-shape graphs: exactly 1. Dynamic: one per bucket (128 / 512 / 2048 / 8192…). |

## Bucket

| Field | Type | Meaning |
|---|---|---|
| `shape_hint` | `Shape` | Concrete shape this bucket is optimized for. |
| `schedule` | `[ScheduleEntry]` | Ordered list of per-Tile kernel dispatches. |

## ScheduleEntry

| Field | Type | Meaning |
|---|---|---|
| `tile_id` | `u32` | 0–63 (physical Tile). 64 / 65 reserved for redundancy. |
| `kernel_index` | `u32` | Index into `Graph.kernels`. |
| `args_offset` | `u64` | Offset into per-dispatch args buffer (future: `Graph.args_buffer`). |
| `args_size` | `u32` | Size of args blob in bytes. |

## Worked example

See `fixtures/minimal_v0_1.npug` + `examples/gen_golden.rs`. Inspect with:

```bash
cargo run --example inspect -- fixtures/minimal_v0_1.npug
```

## Cross-references

- Schema source: `schema/npug.fbs`
- Rust API: `src/lib.rs` (re-exports `builder::GraphBuilder`, `reader::GraphReader`)
- C ABI: `include/npug.h` (generated by cbindgen — Task 11 of plan)
- Version constants: `src/version.rs`
