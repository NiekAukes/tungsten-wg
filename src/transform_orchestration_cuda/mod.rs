use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::{
    orchestrate::{Flatten, model::ShaderDependency},
    spmt::model::PermutationTableInput,
    transform_rcl::sanitize_name,
};

mod builders;
pub mod random;

/// Generates a complete CUDA C++ source file containing a `CudaPipeline_{name}`
/// class for running density-function kernels on the GPU via CUDA.
pub struct CudaOrchestrationCodegen {
    /// Accumulated C++ source code.
    code: String,
}

impl CudaOrchestrationCodegen {
    pub fn new() -> Self {
        Self {
            code: String::with_capacity(16 * 1024),
        }
    }

    /// Generate a self-contained CUDA orchestrator for a single density entry.
    ///
    /// * `name`   – human-readable density name (e.g. `"final_density"`).
    /// * `waves`  – shader dependencies grouped into topologically-sorted waves.
    /// * `target` – the final shader whose output is returned to the caller.
    pub fn convert_single_entry(
        &mut self,
        name: &str,
        waves: &[Vec<ShaderDependency<'_>>],
        target: &ShaderDependency<'_>,
    ) {
        let safe_name = sanitize_name(name);

        // Flatten all shaders in wave order for stable indexing.
        let all_shaders: Vec<&ShaderDependency<'_>> = waves.iter().flat_map(|w| w.iter()).collect();

        // Map from ShaderDependency → index in `all_shaders`.
        let shader_index: HashMap<&ShaderDependency<'_>, usize> = all_shaders
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i))
            .collect();

        let target_idx = shader_index[target];
        let (grid_x, grid_y, grid_z) = target.dimensions;

        // Collect all unique perm tables across all shaders (deduplicated by param name).
        let mut all_perm_tables: Vec<&PermutationTableInput> = Vec::new();
        let mut seen_perm: HashSet<String> = HashSet::new();
        for s in &all_shaders {
            for pt in &s.shader.permutation_tables {
                let pname = builders::perm_table_cuda_param_name(pt);
                if seen_perm.insert(pname) {
                    all_perm_tables.push(pt);
                }
            }
        }

        self.emit_header(&safe_name, grid_x, grid_y, grid_z);
        self.emit_perm_table_helper();
        self.emit_class_open(&safe_name);
        self.emit_private_fields(&all_shaders, &all_perm_tables);
        self.emit_public_open();
        self.emit_constructor(&safe_name, &all_shaders, &all_perm_tables);
        self.emit_destructor(&safe_name, &all_shaders, &all_perm_tables);
        self.emit_run(&safe_name, &all_shaders, waves, target_idx);
        self.emit_class_close();
    }

    /// Return the generated C++ source.
    pub fn finish(self) -> String {
        self.code
    }

    // ── private codegen helpers ──────────────────────────────────────────

    fn emit_header(&mut self, name: &str, gx: i32, gy: i32, gz: i32) {
        let total = gx as i64 * gy as i64 * gz as i64;
        writeln!(
            self.code,
            "// Auto-generated CUDA orchestrator — do not edit"
        )
        .unwrap();
        writeln!(self.code, "#pragma once").unwrap();
        writeln!(self.code).unwrap();
        writeln!(self.code, "#include \"density_function.cu\"").unwrap();
        writeln!(self.code, "#include <cuda_runtime.h>").unwrap();
        writeln!(self.code, "#include <vector>").unwrap();
        writeln!(self.code, "#include <cstdint>").unwrap();
        writeln!(self.code, "#include <cstdio>").unwrap();
        writeln!(self.code).unwrap();
        writeln!(self.code, "static const int GRID_X = {};", gx).unwrap();
        writeln!(self.code, "static const int GRID_Y = {};", gy).unwrap();
        writeln!(self.code, "static const int GRID_Z = {};", gz).unwrap();
        writeln!(
            self.code,
            "static const int TOTAL_ELEMENTS = {}; // {} * {} * {}",
            total, gx, gy, gz
        )
        .unwrap();
        writeln!(
            self.code,
            "static const size_t BUFFER_SIZE = (size_t)TOTAL_ELEMENTS * sizeof(float);"
        )
        .unwrap();
        writeln!(self.code).unwrap();
    }

    /// Emit static C++ helper functions for Minecraft-compatible xoroshiro128++
    /// permutation table generation.
    fn emit_perm_table_helper(&mut self) {
        writeln!(
            self.code,
            "// ─── Minecraft Xoroshiro128++ permutation table helper ───────────────────────"
        )
        .unwrap();
        writeln!(
            self.code,
            "static inline uint64_t _mc_rotl64(uint64_t x, int k) {{"
        )
        .unwrap();
        writeln!(self.code, "    return (x << k) | (x >> (64 - k));").unwrap();
        writeln!(self.code, "}}").unwrap();
        writeln!(self.code).unwrap();
        writeln!(
            self.code,
            "static uint64_t _mc_xoro_next_long(uint64_t* s0, uint64_t* s1) {{"
        )
        .unwrap();
        writeln!(self.code, "    uint64_t l = *s0, m = *s1;").unwrap();
        writeln!(
            self.code,
            "    uint64_t result = _mc_rotl64(l + m, 17) + l;"
        )
        .unwrap();
        writeln!(self.code, "    m ^= l;").unwrap();
        writeln!(self.code, "    *s0 = _mc_rotl64(l, 49) ^ m ^ (m << 21);").unwrap();
        writeln!(self.code, "    *s1 = _mc_rotl64(m, 28);").unwrap();
        writeln!(self.code, "    return result;").unwrap();
        writeln!(self.code, "}}").unwrap();
        writeln!(self.code).unwrap();
        writeln!(
            self.code,
            "static int _mc_xoro_next_int(uint64_t* s0, uint64_t* s1, int bound) {{"
        )
        .unwrap();
        writeln!(
            self.code,
            "    uint64_t bits = (_mc_xoro_next_long(s0, s1) >> 33) * (uint64_t)bound;"
        )
        .unwrap();
        writeln!(self.code, "    uint64_t lo = bits & 0xFFFFFFFFULL;").unwrap();
        writeln!(self.code, "    if (lo < (uint64_t)bound) {{").unwrap();
        writeln!(
            self.code,
            "        uint64_t t = (uint64_t)(-(int64_t)bound) & 0xFFFFFFFFULL;"
        )
        .unwrap();
        writeln!(self.code, "        while (lo < t) {{").unwrap();
        writeln!(
            self.code,
            "            bits = (_mc_xoro_next_long(s0, s1) >> 33) * (uint64_t)bound;"
        )
        .unwrap();
        writeln!(self.code, "            lo = bits & 0xFFFFFFFFULL;").unwrap();
        writeln!(self.code, "        }}").unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code, "    return (int)(bits >> 32);").unwrap();
        writeln!(self.code, "}}").unwrap();
        writeln!(self.code).unwrap();
        writeln!(
            self.code,
            "static double _mc_xoro_next_double(uint64_t* s0, uint64_t* s1) {{"
        )
        .unwrap();
        writeln!(
            self.code,
            "    return (double)(_mc_xoro_next_long(s0, s1) >> 11) * 1.1102230246251565e-16;"
        )
        .unwrap();
        writeln!(self.code, "}}").unwrap();
        writeln!(self.code).unwrap();
        writeln!(
            self.code,
            "// Generate a 256-element permutation table matching Minecraft's PerlinNoiseSampler."
        )
        .unwrap();
        writeln!(
            self.code,
            "// ident_lo/hi and subident_lo/hi are precomputed via MD5(string) at codegen time."
        )
        .unwrap();
        writeln!(self.code, "static void make_perm_table(").unwrap();
        writeln!(self.code, "    int8_t* out, int64_t world_seed,").unwrap();
        writeln!(self.code, "    uint64_t ident_lo,    uint64_t ident_hi,").unwrap();
        writeln!(self.code, "    int64_t  subident_index,").unwrap();
        writeln!(self.code, "    uint64_t subident_lo, uint64_t subident_hi").unwrap();
        writeln!(self.code, ") {{").unwrap();
        writeln!(
            self.code,
            "    uint64_t lo = (uint64_t)world_seed ^ ident_lo ^ subident_lo ^ (uint64_t)(subident_index * 2 + 1);"
        )
        .unwrap();
        writeln!(
            self.code,
            "    uint64_t hi = (uint64_t)(world_seed >> 32) ^ ident_hi ^ subident_hi;"
        )
        .unwrap();
        writeln!(
            self.code,
            "    uint64_t s0 = lo ^ UINT64_C(0x6c62272e07bb0142);"
        )
        .unwrap();
        writeln!(
            self.code,
            "    uint64_t s1 = hi ^ UINT64_C(0x62b821756295c58d);"
        )
        .unwrap();
        writeln!(self.code, "    _mc_xoro_next_double(&s0, &s1); // originX").unwrap();
        writeln!(self.code, "    _mc_xoro_next_double(&s0, &s1); // originY").unwrap();
        writeln!(self.code, "    _mc_xoro_next_double(&s0, &s1); // originZ").unwrap();
        writeln!(self.code, "    int8_t table[256];").unwrap();
        writeln!(
            self.code,
            "    for (int i = 0; i < 256; i++) table[i] = (int8_t)i;"
        )
        .unwrap();
        writeln!(self.code, "    for (int i = 0; i < 256; i++) {{").unwrap();
        writeln!(
            self.code,
            "        int j = i + _mc_xoro_next_int(&s0, &s1, 256 - i);"
        )
        .unwrap();
        writeln!(
            self.code,
            "        int8_t tmp = table[i]; table[i] = table[j]; table[j] = tmp;"
        )
        .unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(
            self.code,
            "    for (int i = 0; i < 256; i++) out[i] = table[i];"
        )
        .unwrap();
        writeln!(self.code, "}}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_class_open(&mut self, name: &str) {
        writeln!(
            self.code,
            "// ============================================================================"
        )
        .unwrap();
        writeln!(self.code, "// CUDA PIPELINE: {}", name).unwrap();
        writeln!(
            self.code,
            "// ============================================================================"
        )
        .unwrap();
        writeln!(self.code, "class CudaPipeline_{} {{", name).unwrap();
        writeln!(self.code, "private:").unwrap();
        writeln!(self.code, "    int3   grid_size;").unwrap();
        writeln!(self.code, "    int    total_elements;").unwrap();
        writeln!(self.code, "    size_t buffer_size;").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_private_fields(
        &mut self,
        shaders: &[&ShaderDependency<'_>],
        perm_tables: &[&PermutationTableInput],
    ) {
        writeln!(self.code, "    // Output buffers (one per kernel)").unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "    float* d_{}_output;", sn).unwrap();
        }
        writeln!(self.code).unwrap();

        if !perm_tables.is_empty() {
            writeln!(
                self.code,
                "    // Permutation tables (deduplicated across all kernels)"
            )
            .unwrap();
            for pt in perm_tables {
                let pn = builders::perm_table_cuda_param_name(pt);
                writeln!(self.code, "    int8_t* d_{};", pn).unwrap();
            }
            writeln!(self.code).unwrap();
        }
    }

    fn emit_public_open(&mut self) {
        writeln!(self.code, "public:").unwrap();
    }

    fn emit_constructor(
        &mut self,
        name: &str,
        shaders: &[&ShaderDependency<'_>],
        perm_tables: &[&PermutationTableInput],
    ) {
        writeln!(
            self.code,
            "    CudaPipeline_{}(int64_t world_seed) {{",
            name
        )
        .unwrap();
        writeln!(
            self.code,
            "        grid_size      = make_int3(GRID_X, GRID_Y, GRID_Z);"
        )
        .unwrap();
        writeln!(self.code, "        total_elements = TOTAL_ELEMENTS;").unwrap();
        writeln!(self.code, "        buffer_size    = BUFFER_SIZE;").unwrap();
        writeln!(self.code).unwrap();

        writeln!(self.code, "        // Allocate output buffers").unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(
                self.code,
                "        cudaMalloc(&d_{sn}_output, buffer_size);"
            )
            .unwrap();
        }
        writeln!(self.code).unwrap();

        if !perm_tables.is_empty() {
            writeln!(
                self.code,
                "        // Allocate and initialize permutation tables from world seed"
            )
            .unwrap();
            for pt in perm_tables {
                let pn = builders::perm_table_cuda_param_name(pt);
                let ident_seed = random::xoroshiro_seed(&pt.ident);
                let (subident_lo, subident_hi) = pt
                    .subident
                    .as_deref()
                    .map(random::xoroshiro_seed)
                    .unwrap_or((0, 0));
                writeln!(self.code, "        {{").unwrap();
                writeln!(self.code, "            int8_t h_table[256];").unwrap();
                writeln!(
                    self.code,
                    "            make_perm_table(h_table, world_seed,"
                )
                .unwrap();
                writeln!(
                    self.code,
                    "                UINT64_C(0x{:016x}), UINT64_C(0x{:016x}), // ident: \"{}\"",
                    ident_seed.0, ident_seed.1, pt.ident
                )
                .unwrap();
                writeln!(self.code, "                INT64_C({}),", pt.subident_index).unwrap();
                writeln!(
                    self.code,
                    "                UINT64_C(0x{:016x}), UINT64_C(0x{:016x})  // subident: {}",
                    subident_lo,
                    subident_hi,
                    pt.subident.as_deref().unwrap_or("(none)")
                )
                .unwrap();
                writeln!(self.code, "            );").unwrap();
                writeln!(
                    self.code,
                    "            cudaMalloc(&d_{pn}, 256 * sizeof(int8_t));"
                )
                .unwrap();
                writeln!(
                    self.code,
                    "            cudaMemcpy(d_{pn}, h_table, 256 * sizeof(int8_t), cudaMemcpyHostToDevice);"
                )
                .unwrap();
                writeln!(self.code, "        }}").unwrap();
            }
            writeln!(self.code).unwrap();
        }

        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_destructor(
        &mut self,
        name: &str,
        shaders: &[&ShaderDependency<'_>],
        perm_tables: &[&PermutationTableInput],
    ) {
        writeln!(self.code, "    ~CudaPipeline_{}() {{", name).unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "        cudaFree(d_{sn}_output);").unwrap();
        }
        for pt in perm_tables {
            let pn = builders::perm_table_cuda_param_name(pt);
            writeln!(self.code, "        cudaFree(d_{pn});").unwrap();
        }
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_run(
        &mut self,
        _name: &str,
        shaders: &[&ShaderDependency<'_>],
        waves: &[Vec<ShaderDependency<'_>>],
        target_idx: usize,
    ) {
        writeln!(
            self.code,
            "    /// Execute the full density pipeline and return the target output."
        )
        .unwrap();
        writeln!(self.code, "    std::vector<float> run(float3 origin) {{").unwrap();
        writeln!(self.code, "        const int BLOCK_SIZE = 256;").unwrap();
        writeln!(self.code).unwrap();

        for (wave_idx, wave) in waves.iter().enumerate() {
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
                let kernel_name = sanitize_name(&dep.shader.name);
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
                    "            {kernel_name}<<<num_blocks, BLOCK_SIZE>>>(\n                make_int3(0, 0, 0), make_int3({dim_x}, {dim_y}, {dim_z}), origin,\n                make_float3({os_x}, {os_y}, {os_z}),\n                make_float3({ps_x}, {ps_y}, {ps_z})"
                )
                .unwrap();

                // Density inputs: output buffers from upstream shaders.
                for input_dep in &dep.shader.inputs {
                    let input_sn = shader_dep_name(input_dep);
                    write!(self.code, ",\n                d_{input_sn}_output").unwrap();
                }

                // Permutation table pointers (in order of the shader's perm table list).
                for pt in &dep.shader.permutation_tables {
                    let pn = builders::perm_table_cuda_param_name(pt);
                    write!(self.code, ",\n                d_{pn}").unwrap();
                }

                // Output buffer.
                writeln!(self.code, ",\n                d_{dep_name}_output\n            );").unwrap();
                writeln!(self.code, "        }}").unwrap();
            }

            writeln!(self.code, "        cudaDeviceSynchronize();").unwrap();
            writeln!(self.code).unwrap();
        }

        // Copy target output back to host.
        let target_sn = shader_dep_name(shaders[target_idx]);
        let (target_dim_x, target_dim_y, target_dim_z) = shaders[target_idx].dimensions;
        let target_total_elements = target_dim_x as i64 * target_dim_y as i64 * target_dim_z as i64;
        writeln!(self.code, "        // Copy target output to host").unwrap();
        writeln!(
            self.code,
            "        std::vector<float> result({target_total_elements});"
        )
        .unwrap();
        writeln!(
            self.code,
            "        cudaMemcpy(result.data(), d_{target_sn}_output, (size_t){target_total_elements} * sizeof(float), cudaMemcpyDeviceToHost);"
        )
        .unwrap();
        writeln!(self.code, "        return result;").unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_class_close(&mut self) {
        writeln!(self.code, "}};").unwrap();
        writeln!(self.code).unwrap();
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
