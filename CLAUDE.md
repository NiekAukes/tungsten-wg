# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tungsten-wg` is a Cargo workspace that compiles Minecraft world-generation density functions into generated source code. The root package `tungsten-wg` (`src/lib.rs`) is a thin facade that re-exports the member crates under `crates/` and provides `compile`. It only does backend compilation. The frontends that parse vanilla worldgen JSON live in a separate crate (`tungsten-mc-compile`), so there is no binary and no end-to-end runner here.

Public entrypoint: `compile(&OptimizedProgram, &mut [&mut dyn Backend]) -> Result<(), CompileError>`. Callers build an `OptimizedProgram`, create the backends they want (e.g. `RustBackend::new(RustConfig::default())`), and pass them in. Each backend keeps its own output (e.g. `RustOutput`), so there is no combined output struct. `compile` builds one `Interface` per backend from the backend's `naming_rules()`, then calls `Backend::generate`. It stops at the first failing backend.

## Commands

```bash
cargo build                     # whole workspace (default feature: rust)
cargo test --workspace
cargo test <name_substring>     # run a single test
cargo doc --open
```

Feature `rust` (default) enables the `tungsten-rust` backend in the facade.

The test coverage is thin. `crates/tungsten-rust/src/model/tests/examples.rs` isn't declared in `model/mod.rs`, so it never compiles. Real validation happens downstream by compiling and running the generated code.

## Workspace layout

Dependencies flow downward; `tungsten-ir` is the base.

| Crate                | Role                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tungsten-ir`        | The IR: `OptimizedProgram` and its model (`ComputeUnit`, `Function`, `Constant`, `EntryPoint`, ...), plus the generational `Arena<T>` / `Ref<T>`. A `Ref` used on the wrong arena panics. Depends on nothing.                                                                                                                                                                                              |
| `tungsten-optimizer` | Passes over `OptimizedProgram`. `perform_inline` returns `InlineStats`. `fuse.rs` is WIP and not exported.                                                                                                                                                                                                                                                                                                 |
| `tungsten-interface` | Backend-agnostic contract. `Interface::build(program, naming_rules)` schedules `Instance`s (keyed by unit, dimensions, scales) into waves and fixes every generated name (kernels, buffers, host-input fields, auto consts). Traits: `DeviceCodegen`, `OrchestrationCodegen`, `Backend` (+ `BackendError`). Also `naming.rs` (`NamingRules`) and `scale.rs`, and `pretty/` (printing helpers, DOT output). |
| `tungsten-codegen`   | Generic code-generation constructs. Currently the text `Printer`.                                                                                                                                                                                                                                                                                                                                          |
| `tungsten-rust`      | The reference backend ("RCL", Rust). `RustBackend` / `RustConfig` / `RustOutput` in `backend.rs`; `RCLCodegen` (device side, implements `DeviceCodegen`) and `TransformRclOrchestration` (orchestration side). Has its own target AST in `model/` and `interned.rs`.                                                                                                                                       |
| `tungsten-java`      | Java 21 backend (spec in `SPEC.md`). `JavaBackend` / `JavaConfig` / `JavaOutput` in `backend.rs`; `JavaCodegen` (device side) and `TransformJavaOrchestration`. Own AST and printer in `model/`. Reuses type-derivation and auto-const helpers from `tungsten-rust`. Feature `java` in the facade (off by default). Smoke test in `src/tests.rs`. |
| `tungsten-legacy`    | The old SPMT IR (arena `Interned<'m, T>` nodes in a `bumpalo::Bump`), its `builder/` API and macros (`body!`, `build!`, `extern_fns!`, `pos3!`, `vec3!`), `dag.rs`, `normalize.rs`, `pretty.rs`. `conversion::convert(&SPMT) -> Result<OptimizedProgram>` lowers it into the new IR.                                                                                                                       |

Crate sources sit directly in the crate root (`crates/tungsten-ir/lib.rs`, `crates/tungsten-interface/lib.rs`, `crates/tungsten-codegen/lib.rs`) for some crates and under `src/` for others (`tungsten-legacy`, `tungsten-optimizer`, `tungsten-rust`). Check each `Cargo.toml` if unsure.

The facade re-exports: `codegen`, `interface`, `ir`, `optimizer`, `spmt` (= `tungsten-legacy`), `rust` (feature-gated), plus the legacy builder macros.

### Pipeline

1. A frontend builds an `OptimizedProgram` directly, or builds legacy SPMT and calls `tungsten_legacy::conversion::convert`.
2. Optionally run `tungsten_optimizer::perform_inline`.
3. `compile` builds the `Interface` and each `Backend::generate` runs its `OrchestrationCodegen` (which drives its `DeviceCodegen`).

In `tungsten-rust`, every density/helper function shares the same leading parameters (`uniform_parameters()`: `p, config, buffers, auto_constants, host_consts`). The generated code assumes runtime types (`Pos3`, `Config`, `iter_3d`, samplers, etc.) that aren't generated here. The crate-level doc in `crates/tungsten-rust/src/lib.rs` lists them. It also mentions an `INSTRUCTIONS.md` that isn't in the repo.

### Other directories

- `old/` holds the pre-redesign single-crate source. It is not part of the workspace and doesn't compile. Only the CUDA pieces (`old/cuda`, `old/transform_cuda`, `old/transform_orchestration_cuda`) still have content, kept as reference for the CUDA rework. The other subdirectories are empty.

Planned backends (`tungsten-cuda`, OpenCL) should follow the `tungsten-rust` structure on top of `tungsten-interface` and be validated against Rust output.

## Conventions

- A new backend is a new crate implementing `Backend` (with its own config and output structs), optionally feature-gated in the root `Cargo.toml` and re-exported from `src/lib.rs`. It doesn't touch `compile`.
- Names that generated code refers to must come from the shared `Interface`, never be recomputed in a backend.
- Shared dependencies go in `[workspace.dependencies]` and are used with `.workspace = true`.
- Each backend that uses AST arenas allocates its own `Bump`. AST lifetimes are tied to that arena.
- Errors go through `BackendError` / `CompileError`, but many failure paths in the backends still `unwrap`/`panic!`.
