# Tungsten Worldgen Compiler

A compiler and code generator for Minecraft world generation algorithms, transforming density functions into optimized Rust and CUDA code.

## Overview

Tungsten Worldgen Compiler takes Minecraft world generation density functions and compiles them into high-performance implementations in multiple target languages. It uses a Single Program Multiple Target (SPMT) intermediate representation to analyze and optimize worldgen algorithms before generating code for different compute backends.

This crate only provides backend compilation and code generation functionality. The end-to-end Minecraft compiler is available as `tungsten-mc-compile`.

## Features

- **Multi-Target Compilation**: Generate code for multiple backends from a single source
  - **Rust**: CPU-optimized Rust code
  - **CUDA C++**: GPU-accelerated CUDA kernels
  - **OpenCL**: Cross-platform GPU compute (in development)

- **Minecraft Integration**: Designed specifically for Minecraft's world generation algorithms with support for vanilla worldgen formats (1.21.1)

- **Visual Debugging**: Generate DOT graph files for visualizing density function dependencies and execution waves

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tungsten-wg = "0.1.0"
```

Or install from GitHub:

```toml
[dependencies]
tungsten-wg = { git = "https://github.com/NiekAukes/tungsten-wg" }
```

## Usage

### Basic Compilation

```rust
use tungsten_wg::{CompilerConfig, Program};

// Load or construct an SPMT program
let program = Program::from_file("worldgen.spmt")?;

// Configure the compiler
let config = CompilerConfig::new()
    .with_rcl(true)      // Enable Rust backend
    .with_cuda(true);    // Enable CUDA backend

// Compile to target languages
let output = program.compile(&config)?;

// Write generated code
std::fs::write("output/density_function.rs", output.rcl_code)?;
std::fs::write("output/kernels.cu", output.cuda_code)?;
```

### Configuration Options

```rust
let config = CompilerConfig {
    generate_rcl: true,                              // Generate Rust code
    generate_cuda: false,                            // Generate CUDA code
    generate_gpu_orchestrator: false,                // Generate GPU orchestrator
    rcl_density_module_name: "density_function".to_string(),
    rcl_orchestration_module_name: "orchestration".to_string(),
};
```

## Architecture

The compiler pipeline consists of several stages:

1. **SPMT Generation**: Parse density function definitions into the SPMT intermediate representation, done by dedicated parsers (not included in this crate)
2. **Orchestration**: Schedule execution waves and manage inter-kernel dependencies
3. **Transformation**: Convert SPMT IR to target-specific ASTs (RCL, CUDA)
4. **Code Generation**: Generate final source code in target languages

## Development

### Building

```bash
cargo build
```

### Running Tests

```bash
cargo test
```

### Generating Documentation

```bash
cargo doc --open
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

https://github.com/NiekAukes/tungsten-wg
