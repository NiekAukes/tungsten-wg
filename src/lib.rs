//! # Tungsten Worldgen Compiler
//!
//! A compiler pipeline for transforming Minecraft world generation algorithms.
//! This library takes a Single Program Multiple Target (SPMT) Intermediate Representation
//! of worldgen density functions and compiles them into optimized
//! target languages, specifically Rust (RCL) and CUDA C++.
//!
//! It handles the orchestration of execution waves, ensuring dependencies between
//! compute shaders/kernels are correctly scheduled and pruned.

#![allow(warnings)]

use bumpalo::Bump;
use std::collections::HashMap;
use thiserror::Error;

// --- Modules ---

/// CUDA models and code generation utilities.
pub mod cuda;
/// Wave orchestration and execution scheduling logic.
pub mod orchestrate;
/// Rust Code Language (RCL) models and inline code generation.
pub mod rcl;
/// Single Program Multiple Thread (SPMT) Intermediate Representation definitions.
pub mod spmt;
/// Transformations from SPMT IR to CUDA AST.
pub mod transform_cuda;
/// Transformations from base orchestration to CUDA-specific orchestration.
pub mod transform_orchestration_cuda;
/// Transformations from base orchestration to generalized GPU compute orchestration.
pub mod transform_orchestration_gpu;
/// Transformations from base orchestration to RCL orchestration.
pub mod transform_orchestration_rcl;
/// Transformations from SPMT IR to RCL AST.
pub mod transform_rcl;

use crate::{
    cuda::{codegen::CudaCodeGenerator, model::CudaModule},
    rcl::codegen::RustCodeGenerator,
    transform_orchestration_cuda::CudaOrchestrationCodegen,
    transform_orchestration_gpu::GpuOrchestrationCodegen,
    transform_orchestration_rcl::OrchestrationConverter,
};

use crate::spmt::model::SPMT as Program;

// --- Error Handling ---

/// Represents errors that can occur during the SPMT compilation process.
#[derive(Debug, Error)]
pub enum CompileError {
    /// Indicates a failure while building or scheduling the execution wave graph.
    #[error("Failed to generate Orchestration code")]
    OrchestrationError,
    /// Indicates a failure during the CUDA code generation phase.
    #[error("Failed to generate CUDA code")]
    CudaError,
}

/// A specialized `Result` type for compilation operations.
pub type Result<T> = std::result::Result<T, CompileError>;

// --- Configuration & Builder ---

/// Configuration builder for the code generation step.
///
/// Use this to toggle specific backends (RCL, CUDA) and override
/// the default module names generated for the Rust outputs.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Whether to generate the Rust (RCL) backend.
    pub generate_rcl: bool,
    /// Whether to generate the generalized GPU orchestrator (currently disabled).
    pub generate_gpu_orchestrator: bool,
    /// Whether to generate the CUDA C++ backend.
    pub generate_cuda: bool,
    /// The namespace to use for the RCL density function module.
    pub rcl_density_module_name: String,
    /// The namespace to use for the RCL orchestration module.
    pub rcl_orchestration_module_name: String,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            generate_rcl: true,
            generate_gpu_orchestrator: false,
            generate_cuda: false,
            rcl_density_module_name: "density_function".to_string(),
            rcl_orchestration_module_name: "orchestration".to_string(),
        }
    }
}

impl CompilerConfig {
    /// Creates a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggles the generation of the Rust Code Layer (RCL) backend.
    pub fn with_rcl(mut self, generate: bool) -> Self {
        self.generate_rcl = generate;
        self
    }

    // pub fn with_gpu_orchestrator(mut self, generate: bool) -> Self {
    //     self.generate_gpu_orchestrator = generate;
    //     self
    // }

    /// Toggles the generation of the CUDA C++ backend.
    pub fn with_cuda(mut self, generate: bool) -> Self {
        self.generate_cuda = generate;
        self
    }

    /// Overrides the generated module names for the RCL backend.
    ///
    /// # Arguments
    /// * `density` - The name for the module containing mathematical density logic.
    /// * `orchestration` - The name for the module handling wave dispatch.
    pub fn rcl_module_names(mut self, density: &str, orchestration: &str) -> Self {
        self.rcl_density_module_name = density.to_string();
        self.rcl_orchestration_module_name = orchestration.to_string();
        self
    }
}

// --- Output Structure ---

/// Holds the generated raw source code strings for whichever backends were enabled
/// in the [`CompilerConfig`].
#[derive(Debug, Default)]
pub struct CompiledOutput {
    /// The concatenated Rust source code.
    /// Contains both the density function mathematical logic and the wave orchestration logic.
    pub rcl: Option<String>,
    //pub rcl_orchestration: Option<String>,
    //pub gpu_orchestrator: Option<String>,
    /// The generated CUDA device C++ code containing the mathematical density kernels.
    pub cuda_density_function: Option<String>,
    /// The generated CUDA host C++ code handling memory and kernel launch orchestration.
    pub cuda_orchestration: Option<String>,
}

// --- Main API Entrypoint ---

/// Compiles the provided SPMT program into the target language backends.
///
/// # Arguments
/// * `program` - A reference to the parsed SPMT Intermediate Representation.
/// * `config` - Defines which languages to compile to and how to format them.
///
/// # Returns
/// A [`CompiledOutput`] struct containing `Some(String)` for every enabled backend,
/// or a [`CompileError`] if AST transformation fails.
pub fn compile(program: &Program, config: &CompilerConfig) -> Result<CompiledOutput> {
    let mut output = CompiledOutput::default();

    // Generate Base Orchestration DAG (Shared across multiple backends)
    let orchestration_arena = Bump::new();
    let orchestration = orchestrate::transform::transform_from_spmt(program, &orchestration_arena);
    let waves = orchestration.arrange_waves();

    // 1. Generate Rust
    if config.generate_rcl {
        let mut orchestration_conv = OrchestrationConverter::new(&orchestration_arena);
        orchestration_conv.convert(&waves, orchestration.get_primary_shaders());

        // Generate a pruned orchestration function for each primary density
        for primary in &orchestration.get_primary_shaders() {
            let name = &primary.shader.name;
            let pruned_waves = orchestration.arrange_waves_for(primary);
            orchestration_conv.convert_single_entry(name, pruned_waves, primary);
        }

        let orchestration_rcl = orchestration_conv.finish();
        let orch_output = RustCodeGenerator
            .generate_inline_module(&orchestration_rcl, &config.rcl_orchestration_module_name);

        let rcl_model =
            transform_rcl::convert_spmt_to_inline_rcl(program, &waves, &orchestration_arena);
        let rcl_output =
            RustCodeGenerator.generate_inline_module(&rcl_model, &config.rcl_density_module_name);

        // Concatenate density functions and orchestration into a single generated Rust source
        output.rcl = Some(rcl_output + "\n\n" + &orch_output);
    }

    // 2. Generate GPU Orchestrator
    // if config.generate_gpu_orchestrator {
    //     let mut gpu_codegen = GpuOrchestrationCodegen::new();
    //     for primary in &orchestration.get_primary_shaders() {
    //         let name = &primary.shader.name;
    //         let pruned_waves = orchestration.arrange_waves_for(primary);
    //         gpu_codegen.convert_single_entry(name, &pruned_waves, primary);
    //     }
    //     output.gpu_orchestrator = Some(gpu_codegen.finish());
    // }

    // 3. Generate CUDA
    if config.generate_cuda {
        let cuda_arena = Bump::new();

        // 3a. CUDA Density Function Module
        let mut cuda_module = CudaModule::new();
        cuda_module.add_include("\"helpers.cu\"".to_string());

        for density_function in &program.density_functions {
            transform_cuda::add_density_to_cuda_module(
                &mut cuda_module,
                density_function,
                &cuda_arena,
                HashMap::new(),
            );
        }
        let cuda_generator = CudaCodeGenerator::new();
        output.cuda_density_function = Some(cuda_generator.generate_module(&cuda_module));

        // 3b. CUDA Orchestration Module
        let mut cuda_orchestration_codegen = CudaOrchestrationCodegen::new();

        for primary in &orchestration.get_primary_shaders() {
            let pruned_waves = orchestration.arrange_waves_for(primary);
            cuda_orchestration_codegen.convert_single_entry(
                &primary.shader.name,
                pruned_waves.as_ref(),
                primary,
            );
        }
        output.cuda_orchestration = Some(cuda_orchestration_codegen.finish());
    }

    Ok(output)
}
