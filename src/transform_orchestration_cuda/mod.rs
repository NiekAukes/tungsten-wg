use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::spmt::normalize::NormalizedShaderInput;
use crate::{
    orchestrate::{Flatten, model::ShaderDependency},
    spmt::model::PermutationTableInput,
    transform_rcl::sanitize_name,
};

mod builders;
pub mod random;

/// Entry point data collected from a single call to `convert_single_entry`.
#[derive(Clone)]
struct EntryPoint<'a> {
    name: String,
    shaders: Vec<ShaderDependency<'a>>,
    waves: Vec<Vec<ShaderDependency<'a>>>,
    target_idx: usize,
}

/// Generates a complete CUDA C++ source file containing a single `CudaPipeline`
/// class for running all density-function entry points on the GPU via CUDA.
/// All permutation tables are loaded upfront, and each entry point has its own
/// method (e.g. `run_final_density()`, `run_temperature()`, etc.).
pub struct CudaOrchestrationCodegen<'a> {
    /// Accumulated C++ source code.
    code: String,
    /// Collected entry points (name, shaders, waves, target index).
    entry_points: Vec<EntryPoint<'a>>,
    /// All unique permutation tables across all entry points.
    all_perm_tables: Vec<PermutationTableInput>,
}

impl<'a> CudaOrchestrationCodegen<'a> {
    pub fn new() -> Self {
        Self {
            code: String::with_capacity(32 * 1024),
            entry_points: Vec::new(),
            all_perm_tables: Vec::new(),
        }
    }

    /// Collect data for a single density entry point.
    ///
    /// * `name`   – human-readable density name (e.g. `"final_density"`).
    /// * `waves`  – shader dependencies grouped into topologically-sorted waves.
    /// * `target` – the final shader whose output is returned to the caller.
    pub fn convert_single_entry(
        &mut self,
        name: &str,
        waves: Vec<Vec<ShaderDependency<'a>>>,
        target: &ShaderDependency<'a>,
    ) {
        // Flatten all shaders in wave order for stable indexing.
        let all_shaders: Vec<ShaderDependency<'a>> = waves.iter().flat_map(|w| w.iter().cloned()).collect();

        // Map from ShaderDependency → index in `all_shaders`.
        let shader_index: HashMap<&ShaderDependency<'a>, usize> = all_shaders
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();

        let target_idx = shader_index[target];

        // Collect unique perm tables for this entry point and add to global set.
        let mut seen_perm: HashSet<String> = self.all_perm_tables
            .iter()
            .map(|pt| builders::perm_table_cuda_param_name(pt))
            .collect();

        for s in all_shaders.clone() {
            for pt in s.shader.permutation_tables.clone() {
                let pname = builders::perm_table_cuda_param_name(&pt);
                if seen_perm.insert(pname) {
                    self.all_perm_tables.push(pt);
                }
            }
        }

        self.entry_points.push(EntryPoint {
            name: name.to_string(),
            shaders: all_shaders,
            waves: waves.to_vec(),
            target_idx,
        });
    }

    /// Return the generated C++ source with a single unified CudaPipeline class.
    pub fn finish(mut self) -> String {
        if self.entry_points.is_empty() {
            return self.code;
        }

        // Sort permutation tables for stable ordering
        self.all_perm_tables.sort();

        self.emit_header();
        self.emit_buffer_sizes();
        self.emit_class_open();
        self.emit_private_fields();
        self.emit_public_open();
        self.emit_constructor();
        self.emit_destructor();
        self.emit_entry_point_methods();
        self.emit_class_close();

        self.code
    }

    // ── private codegen helpers ──────────────────────────────────────────

    fn emit_header(&mut self) {
        writeln!(
            self.code,
            "// Auto-generated CUDA orchestrator — do not edit"
        )
        .unwrap();
        writeln!(self.code, "#pragma once").unwrap();
        writeln!(self.code).unwrap();
        writeln!(self.code, "#include \"density_function.cu\"").unwrap();
        writeln!(self.code, "#include \"helpers.cu\"").unwrap();
        writeln!(self.code, "#include <cuda_runtime.h>").unwrap();
        writeln!(self.code, "#include <vector>").unwrap();
        writeln!(self.code, "#include <cstdint>").unwrap();
        writeln!(self.code, "#include <cstdio>").unwrap();
        writeln!(self.code, "#include <map>").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_buffer_sizes(&mut self) {
        writeln!(self.code, "    // Buffer sizes for each unique shader output").unwrap();
        /*
        let target_shader = &ep.shaders[ep.target_idx];
        let target_sn = shader_dep_name(target_shader);
        let (target_dim_x, target_dim_y, target_dim_z) = target_shader.dimensions;
        let target_total_elements = target_dim_x as i64 * target_dim_y as i64 * target_dim_z as i64;
        writeln!(self.code, "        // Copy target output to host").unwrap();
        writeln!(
            self.code,
            "        std::vector<double> result({target_total_elements});"
        )
        .unwrap();
         */
        for ep in self.entry_points.iter() {
            let target_shader = &ep.shaders[ep.target_idx];
            let target_sn = target_shader.shader.name.clone();
            let (target_dim_x, target_dim_y, target_dim_z) = target_shader.dimensions;
            let target_total_elements = target_dim_x as i64 * target_dim_y as i64 * target_dim_z as i64;
            writeln!(self.code, "const int {target_sn}_total_elements = {target_total_elements};").unwrap();
            writeln!(self.code, "const int {target_sn}_dim_x = {target_dim_x};").unwrap();
            writeln!(self.code, "const int {target_sn}_dim_y = {target_dim_y};").unwrap();
            writeln!(self.code, "const int {target_sn}_dim_z = {target_dim_z};").unwrap();
        }
    }

    fn emit_class_open(&mut self) {
        writeln!(
            self.code,
            "// ============================================================================"
        )
        .unwrap();
        writeln!(self.code, "// CUDA PIPELINE: Unified entry point orchestrator").unwrap();
        writeln!(
            self.code,
            "// ============================================================================"
        )
        .unwrap();
        writeln!(self.code, "class CudaPipeline {{").unwrap();
        writeln!(self.code, "private:").unwrap();
        writeln!(
            self.code,
            "    cudaStream_t stream; // dedicated stream so instances run concurrently"
        )
        .unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_private_fields(&mut self) {
        // Collect all unique shaders across all entry points
        let mut seen_shaders: HashSet<String> = HashSet::new();
        let mut unique_shaders: Vec<(usize, String)> = Vec::new();

        for ep in &self.entry_points {
            for (idx, shader) in ep.shaders.iter().enumerate() {
                let sn = shader_dep_name(shader);
                if seen_shaders.insert(sn.clone()) {
                    unique_shaders.push((idx, sn));
                }
            }
        }

        writeln!(self.code, "    // Output buffers (one per unique shader across all entry points)").unwrap();
        for (_, sn) in &unique_shaders {
            writeln!(self.code, "    double* d_{}_output;", sn).unwrap();
        }
        writeln!(self.code).unwrap();

        if !self.all_perm_tables.is_empty() {
            writeln!(
                self.code,
                "    // Permutation tables (loaded once for all entry points)"
            )
            .unwrap();
            for pt in &self.all_perm_tables {
                let pn = builders::perm_table_cuda_param_name(pt);
                writeln!(self.code, "    int8_t* d_{};", pn).unwrap();
            }
            writeln!(self.code).unwrap();
        }
    }

    fn emit_public_open(&mut self) {
        writeln!(self.code, "public:").unwrap();
    }

    fn emit_constructor(&mut self) {
        writeln!(self.code, "    CudaPipeline(int64_t world_seed) {{").unwrap();
        writeln!(self.code, "        cudaStreamCreate(&stream);").unwrap();
        writeln!(self.code).unwrap();

        // Collect all unique shaders
        let mut seen_shaders: HashSet<String> = HashSet::new();
        writeln!(self.code, "        // Allocate output buffers for all unique shaders").unwrap();
        for ep in &self.entry_points {
            for shader in &ep.shaders {
                let sn = shader_dep_name(shader);
                if seen_shaders.insert(sn.clone()) {
                    let (dim_x, dim_y, dim_z) = shader.dimensions;
                    let buffer_sz = dim_x as i64 * dim_y as i64 * dim_z as i64;
                    writeln!(
                        self.code,
                        "        cudaMalloc(&d_{sn}_output, (size_t){} * sizeof(double));",
                        buffer_sz
                    )
                    .unwrap();
                }
            }
        }
        writeln!(self.code).unwrap();

        if !self.all_perm_tables.is_empty() {
            writeln!(
                self.code,
                "        // Allocate and initialize permutation tables from world seed"
            )
            .unwrap();
            // Collect perm tables into a separate vec to avoid borrow conflicts
            let perm_tables: Vec<PermutationTableInput> = self.all_perm_tables.clone();
            for pt in perm_tables {
                match pt {
                    PermutationTableInput::PerlinNoise { .. } => self.emit_perm_table_perlin(&pt),
                    PermutationTableInput::Base3DNoise => self.emit_perm_table_base3d(),
                }
            }
            writeln!(self.code).unwrap();
        }

        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_destructor(&mut self) {
        writeln!(self.code, "    ~CudaPipeline() {{").unwrap();

        // Collect all unique shaders
        let mut seen_shaders: HashSet<String> = HashSet::new();
        for ep in &self.entry_points {
            for shader in &ep.shaders {
                let sn = shader_dep_name(shader);
                if seen_shaders.insert(sn.clone()) {
                    writeln!(self.code, "        cudaFree(d_{sn}_output);").unwrap();
                }
            }
        }

        for pt in &self.all_perm_tables {
            let pn = builders::perm_table_cuda_param_name(pt);
            writeln!(self.code, "        cudaFree(d_{pn});").unwrap();
        }

        writeln!(self.code, "        cudaStreamDestroy(stream);").unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_entry_point_methods(&mut self) {
        // Collect entry points into a separate vec to avoid borrow conflicts
        let entry_points: Vec<EntryPoint<'a>> = self.entry_points.clone();
        for ep in entry_points {
            self.emit_single_run_method(&ep);
        }
    }

    fn emit_single_run_method(&mut self, ep: &EntryPoint<'_>) {
        let safe_name = sanitize_name(&ep.name);
        writeln!(
            self.code,
            "    /// Execute the {} pipeline and return the target output.",
            ep.name
        )
        .unwrap();
        writeln!(
            self.code,
            "    void run_{}(double3 origin, double* output) {{",
            safe_name
        )
        .unwrap();
        writeln!(self.code, "        const int BLOCK_SIZE = 256;").unwrap();
        writeln!(self.code).unwrap();

        for (wave_idx, wave) in ep.waves.iter().enumerate() {
            let wave_names: Vec<String> =
                wave.iter().map(|s| sanitize_name(&s.shader.name)).collect();
            writeln!(
                self.code,
                "        // Wave {}: {}",
                wave_idx,
                wave_names.join(", ")
            )
            .unwrap();

            for dep in wave {
                // let kernel_name = sanitize_name(&dep.shader.name);
                let kernel_name = builders::density_function_cuda_name(&dep);
                let dep_name = shader_dep_name(dep);
                let (dim_x, dim_y, dim_z) = dep.dimensions;
                let total_elements_for_shader = dim_x as i64 * dim_y as i64 * dim_z as i64;

                writeln!(
                    self.code,
                    "        {{ // Kernel {kernel_name} with dimensions {dim_x}x{dim_y}x{dim_z}"
                )
                .unwrap();
                writeln!(
                    self.code,
                    "            int num_blocks = ({total_elements_for_shader} + BLOCK_SIZE - 1) / BLOCK_SIZE;"
                )
                .unwrap();

                let (os_x, os_y, os_z) = dep.scaled_origin.as_float();
                let (ps_x, ps_y, ps_z) = dep.scaled_position.as_float();
                write!(
                    self.code,
                    "            {kernel_name}<<<num_blocks, BLOCK_SIZE, 0, stream>>>(\n                make_int3(0, 0, 0), make_int3({dim_x}, {dim_y}, {dim_z}), origin,\n                make_double3({os_x}, {os_y}, {os_z}),\n                make_double3({ps_x}, {ps_y}, {ps_z})"
                )
                .unwrap();

                // Density inputs: output buffers from upstream shaders.
                for input_dep in &dep.normal_shader_dependency_list() {
                    let input_sn = shader_dep_name(input_dep);
                    write!(self.code, ",\n                d_{input_sn}_output").unwrap();
                }

                // Permutation table pointers (in order of the shader's perm table list).
                for pt in &dep.shader.permutation_tables {
                    let pn = builders::perm_table_cuda_param_name(pt);
                    write!(self.code, ",\n                d_{pn}").unwrap();
                }

                // Output buffer.
                writeln!(
                    self.code,
                    ",\n                d_{dep_name}_output\n            );"
                )
                .unwrap();
                writeln!(self.code, "        }}").unwrap();
            }

            // writeln!(self.code, "        cudaStreamSynchronize(stream);").unwrap();
            writeln!(self.code).unwrap();
        }

        // Copy target output back to host.
        let target_shader = &ep.shaders[ep.target_idx];
        let target_sn = shader_dep_name(target_shader);
        let (target_dim_x, target_dim_y, target_dim_z) = target_shader.dimensions;
        let target_total_elements = target_dim_x as i64 * target_dim_y as i64 * target_dim_z as i64;
        writeln!(self.code, "        // Copy target output to host").unwrap();
        writeln!(
            self.code,
            "        cudaMemcpyAsync(output, d_{target_sn}_output, (size_t){target_total_elements} * sizeof(double), cudaMemcpyDeviceToHost, stream);"
        )
        .unwrap();
        // writeln!(self.code, "        cudaStreamSynchronize(stream);").unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_class_close(&mut self) {
        writeln!(self.code, "}};").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_perm_table_perlin(&mut self, pt: &PermutationTableInput) {
        let pn = builders::perm_table_cuda_param_name(pt);
        let PermutationTableInput::PerlinNoise {
            ident,
            subident,
            subident_index,
        } = pt
        else {
            panic!("emit_perm_table_perlin called with non-Perlin permutation table");
        };
        let subident_name = subident.as_deref().unwrap_or("");
        let ident_name = ident.as_str();
        let ident_seed = random::xoroshiro_seed(&ident);
        let (subident_lo, subident_hi) = subident
            .as_deref()
            .map(random::xoroshiro_seed)
            .unwrap_or((0, 0));
        writeln!(self.code, "        {{").unwrap();
        writeln!(self.code, "            PerlinNoiseGenerator pns;").unwrap();
        writeln!(
            self.code,
            "            make_perm_table(&pns, world_seed,"
        )
        .unwrap();
        writeln!(
            self.code,
            "                INT64_C(0x{:016x}), INT64_C(0x{:016x}), // ident: \"{}\"",
            ident_seed.0, ident_seed.1, ident_name
        )
        .unwrap();
        writeln!(self.code, "                INT64_C({}),", subident_index).unwrap();
        writeln!(
            self.code,
            "                INT64_C(0x{:016x}), INT64_C(0x{:016x})  // subident: {}",
            subident_lo, subident_hi, subident_name
        )
        .unwrap();
        writeln!(self.code, "            );").unwrap();
        writeln!(
            self.code,
            "            cudaMalloc(&d_{pn}, sizeof(PerlinNoiseGenerator));"
        )
        .unwrap();
        writeln!(
            self.code,
            "            cudaMemcpy(d_{pn}, &pns, sizeof(PerlinNoiseGenerator), cudaMemcpyHostToDevice);"
        )
        .unwrap();
        writeln!(self.code, "        }}").unwrap();
    }

    fn emit_perm_table_base3d(&mut self) {
        let pn = builders::perm_table_cuda_param_name(&PermutationTableInput::Base3DNoise);
        writeln!(self.code, "        {{").unwrap();
        writeln!(self.code, "            InterpolatedNoiseSamplerGPU pns;").unwrap();
        writeln!(
            self.code,
            "            create_base3d(&pns, world_seed);"
        )
        .unwrap();
        writeln!(
            self.code,
            "            cudaMalloc(&d_{pn}, sizeof(InterpolatedNoiseSamplerGPU));"
        )
        .unwrap();
        writeln!(
            self.code,
            "            cudaMemcpy(d_{pn}, &pns, sizeof(InterpolatedNoiseSamplerGPU), cudaMemcpyHostToDevice);"
        )
        .unwrap();
        writeln!(self.code, "        }}").unwrap();
    }
}

fn shader_dep_name(dep: &ShaderDependency<'_>) -> String {
    sanitize_name(&format!(
        "{}_d{}x{}x{}os{}x{}x{}ps{}x{}x{}",
        dep.shader.name,
        dep.dimensions.0,
        dep.dimensions.1,
        dep.dimensions.2,
        dep.scaled_origin.as_int().0,
        dep.scaled_origin.as_int().1,
        dep.scaled_origin.as_int().2,
        dep.scaled_position.as_int().0,
        dep.scaled_position.as_int().1,
        dep.scaled_position.as_int().2,
    ))
}
