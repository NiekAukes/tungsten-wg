#![allow(warnings)]

use bumpalo::Bump;
use std::collections::HashMap;
use thiserror::Error;

// --- Modules ---
pub mod cuda;
pub mod orchestrate;
pub mod rcl;
pub mod spmt;
pub mod transform_cuda;
pub mod transform_orchestration_cuda;
pub mod transform_orchestration_gpu;
pub mod transform_orchestration_rcl;
pub mod transform_rcl;

use crate::{
    cuda::{codegen::CudaCodeGenerator, model::CudaModule},
    rcl::codegen::RustCodeGenerator,
    transform_orchestration_cuda::CudaOrchestrationCodegen,
    transform_orchestration_gpu::GpuOrchestrationCodegen,
    transform_orchestration_rcl::OrchestrationConverter,
};

// Assuming `Program` is defined in your `spmt` module
use crate::spmt::model::SPMT as Program;

// --- Error Handling ---
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("Failed to generate Orchestration code")]
    OrchestrationError,
    #[error("Failed to generate CUDA code")]
    CudaError,
}

pub type Result<T> = std::result::Result<T, CompileError>;

// --- Configuration & Builder ---

/// Configuration options for the code generation step.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub generate_rcl: bool,
    pub generate_gpu_orchestrator: bool,
    pub generate_cuda: bool,
    pub rcl_density_module_name: String,
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_rcl(mut self, generate: bool) -> Self {
        self.generate_rcl = generate;
        self
    }

    // pub fn with_gpu_orchestrator(mut self, generate: bool) -> Self {
    //     self.generate_gpu_orchestrator = generate;
    //     self
    // }

    pub fn with_cuda(mut self, generate: bool) -> Self {
        self.generate_cuda = generate;
        self
    }

    pub fn rcl_module_names(mut self, density: &str, orchestration: &str) -> Self {
        self.rcl_density_module_name = density.to_string();
        self.rcl_orchestration_module_name = orchestration.to_string();
        self
    }
}

// --- Output Structure ---

/// Holds the generated code strings for whichever backends were enabled.
#[derive(Debug, Default)]
pub struct CompiledOutput {
    pub rcl: Option<String>,
    //pub rcl_orchestration: Option<String>,
    //pub gpu_orchestrator: Option<String>,
    pub cuda_density_function: Option<String>,
    pub cuda_orchestration: Option<String>,
}

// --- Main API Entrypoint ---

/// Compiles the provided SPMT program into the configured target languages.
pub fn compile(program: &Program, config: &CompilerConfig) -> Result<CompiledOutput> {
    let mut output = CompiledOutput::default();

    // Generate Base Orchestration DAG (Shared across multiple backends)
    let orchestration_arena = Bump::new();
    let orchestration = orchestrate::transform::transform_from_spmt(program, &orchestration_arena);
    let waves = orchestration.arrange_waves();

    // 1. Generate RCL (Rust Code Layer)
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

        //output.rcl_orchestration = Some(orch_output);
        //output.rcl_density_function = Some(rcl_output);
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
