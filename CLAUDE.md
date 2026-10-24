# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tungsten-wg` is a library-only crate (edition 2024) that compiles Minecraft world-generation density functions, given as an SPMT IR, into generated source code (Rust, called "RCL", and CUDA C++). It only does backend compilation. The frontends that parse vanilla worldgen JSON into SPMT live in a separate crate (`tungsten-mc-compile`), so this repo has no binary and no end-to-end runner. The single public entrypoint is `compile(&SPMT, &CompilerConfig) -> CompiledOutput` in `src/lib.rs`.

## Commands

```bash
cargo build
cargo test                      # currently runs 0 tests (see below)
cargo test <name_substring>     # run a single test
cargo doc --open
```

`#![allow(warnings)]` is set crate-wide, so the compiler shows no warnings. Don't expect lint feedback from `cargo build`.

The test coverage is thin. `src/rcl/tests/examples.rs` exists but isn't declared in `src/rcl/mod.rs`, so it never compiles. The tests in `src/spmt_optimize/conversion.rs` are commented out. Real validation happens downstream by compiling and running the generated code.

## Architecture

There are two parallel pipelines. The repo is partway through migrating from the old one to the new one (see `TODO.md`).

### Legacy pipeline (SPMT → orchestration DAG → backends)

1. **`spmt/`**: the source IR. `SPMT` holds `DensityFunction`s (compute units with `density_inputs`, `permutation_table_inputs`, `host_inputs`, a statement body, and helper functions) plus `main_density_functions`, which are the entry points with dimensions and scales. Nodes are `Interned<'m, T>` refs allocated in a `bumpalo::Bump` arena. `spmt/builder/` (with `macros.rs`) is the ergonomic API frontends use to construct SPMT.
2. **`orchestrate/`**: `transform_from_spmt` builds a shader dependency DAG. `arrange_waves()` groups shaders into dependency-ordered execution waves, and `arrange_waves_for(primary)` returns the waves pruned to one entry point. `Scale` uses fixed-point hashing so float scales can serve as dedup keys.
3. **Backends**: each backend pairs a target AST/model with a code generator:
   - `transform_rcl` (SPMT → `rcl::model`) + `transform_orchestration_rcl` → `rcl::codegen::RustCodeGenerator`
   - `transform_cuda` + `transform_orchestration_cuda` → `cuda::codegen::CudaCodeGenerator`. The generated code `#include`s a runtime `helpers.cu` that this repo doesn't contain.
   - `transform_orchestration_gpu` is currently disabled in `lib.rs`.

### New pipeline (SPMT → OptimizedProgram → uniform-signature backends)

1. **`spmt_optimize/`**: `conversion::convert(&SPMT)` lowers to `OptimizedProgram`, which uses its own generational arena (`Arena<T>` / `Ref<T>` with an owner tag; using a `Ref` on the wrong arena panics). `perform_inline` then runs the inliner. `fuse.rs` is WIP.
2. **`interface/`**: the backend-agnostic contract.
   - `Instance` is one scheduled use of a compute unit, keyed by (unit, dimensions, scales).
   - `Interface` fixes every generated name (kernels, buffers, host-input fields, auto consts) once, so callers and callees agree. Naming helpers live in `interface/naming.rs`.
   - Traits: `DeviceCodegen` (per-language device code) and `OrchestrationCodegen` (the glue that runs kernels in wave order).
3. **`transform_rcl_new/`**: the reference implementation of these traits. `RCLCodegen` is the device side and `TransformRclOrchestration` is the orchestration side. Every density/helper function shares the same leading parameters (`uniform_parameters()`: `p, config, buffers, auto_constants, host_consts`). The generated code assumes runtime types (`Pos3`, `Config`, `iter_3d`, samplers, etc.) that aren't generated here. The module doc lists them. The doc also mentions an `INSTRUCTIONS.md` that isn't in the repo.

Planned backends (CUDA rework, Java, OpenCL) are meant to follow the `transform_rcl_new` structure on top of `interface` and be validated against RCL output (see `TODO.md`).

### Dead code

`src/transform_naga/`, `src/transform_spmt_to_opencl/` and `src/opencl/` aren't declared in `lib.rs` and don't compile into the crate.

## Conventions

- New backend config needs three changes: a flag on `CompilerConfig` (plus a `with_*` builder method), an `Option<String>` field on `CompiledOutput`, and a branch in `compile()`.
- Each backend allocates its own `Bump` arena inside `compile()`. AST lifetimes are tied to that arena.
- `CompileError` exists, but most failure paths currently `unwrap`/`panic!`.
