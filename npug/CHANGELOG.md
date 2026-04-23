# npug changelog

## v0.1.0 — 2026-04-18 (ABI freeze candidate)

Initial public ABI. Schema: `schema/npug.fbs`. Spec: `docs/npug-abi-v0.1.md`.

### Included
- `Graph` root with `abi_version`, `target`, `producer`.
- `Tensor` with `DType` (13 variants: INT/MX/NV/FP), `Shape` (symbolic-capable), `QuantInfo`
  (None / PerTensor / PerChannel / MxBlock32 / NvBlock16), `MemoryRegion`
  (LSRAM / CSRAM / SLC / LPDDR) + offset.
- `Buffer` — raw byte blobs.
- `Kernel` — Tile μC / Sched MCU RISC-V binaries with entry offset.
- `EntryPoint` / `Bucket` / `ScheduleEntry` — per-Tile dispatch schedule, supports dynamic-shape buckets.

### Rust API
- `npug::builder::GraphBuilder`
- `npug::reader::GraphReader` + `TensorView` / `KernelView` / `EntryPointView`
- `npug::version::{CURRENT, MAGIC, is_compatible}`

### C ABI
- `npug_abi_version()`, `npug_validate(bytes, len)`.

### Python
- Stock `flatbuffers` package via `flatc --python`. Verified roundtrip against Rust-produced fixture.

### Tests
- 15 tests across 8 test binaries (roundtrip × 6 + version compat + golden hash + C ABI smoke).
- Golden fixture: `fixtures/minimal_v0_1.npug` (SHA-256 pinned in `tests/golden_fixture.rs`).
- CI: fmt + clippy + test + golden diff + Python smoke.

### Next (v0.1.1+, non-breaking)
- Args buffer for kernel dispatches
- Optional metadata (compile flags, autotune choices)
- Compression flag on `Buffer`

### Breaking (v0.2 / v1.0 — reserved)
- Make `Tensor.quant` skip emission when scheme=None (builder-only size optimization).
- Reconsider `MemoryRegion` placement (per-Tensor vs. per-Buffer).
- Any change here requires coordinated compiler+runtime bump.
