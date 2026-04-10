use std::collections::HashMap;
use std::fmt::Write;

use crate::{
    orchestrate::{Flatten, model::ShaderDependency},
    transform_rcl::sanitize_name,
};

mod builders;
mod output;
mod permutation_tables;
pub mod random;

/// Generates a complete Rust source file containing a `GpuOrchestrator` struct
/// and its `impl` block for running density-function compute shaders on the GPU
/// via wgpu.
pub struct GpuOrchestrationCodegen {
    /// Accumulated Rust source code.
    code: String,
}

impl GpuOrchestrationCodegen {
    pub fn new() -> Self {
        Self {
            code: String::with_capacity(16 * 1024),
        }
    }

    /// Generate a self-contained GPU orchestrator for a single density entry.
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

        // Dimensions of output grid (assumed uniform across shaders).
        let (grid_x, grid_y, grid_z) = target.dimensions;

        self.emit_header(&safe_name, grid_x, grid_y, grid_z);
        self.emit_struct(&safe_name, &all_shaders);
        self.emit_impl_open(&safe_name);
        self.emit_new(&safe_name, &all_shaders, &shader_index, waves);
        self.emit_orchestrate(&safe_name, &all_shaders, &shader_index, waves, target_idx);
        self.emit_impl_close();
        self.emit_helpers();
    }

    /// Return the generated Rust source.
    pub fn finish(self) -> String {
        self.code
    }

    // ── private codegen helpers ──────────────────────────────────────────

    fn emit_header(&mut self, _name: &str, gx: i32, gy: i32, gz: i32) {
        let total = gx as u64 * gy as u64 * gz as u64;
        writeln!(
            self.code,
            "// Auto-generated GPU orchestrator — do not edit"
        )
        .unwrap();
        writeln!(self.code, "#![allow(warnings)]").unwrap();
        writeln!(self.code).unwrap();
        writeln!(self.code, "use crate::mathf64::Vec3;").unwrap();
        writeln!(self.code, "use crate::orchestration::PermutationTables;").unwrap();
        writeln!(self.code, "use crate::utils::PerlinNoiseSampler;").unwrap();
        writeln!(self.code, "use wgpu::util::DeviceExt;").unwrap();
        writeln!(self.code).unwrap();
        writeln!(self.code, "const GRID_X: u32 = {};", gx).unwrap();
        writeln!(self.code, "const GRID_Y: u32 = {};", gy).unwrap();
        writeln!(self.code, "const GRID_Z: u32 = {};", gz).unwrap();
        writeln!(
            self.code,
            "const TOTAL_ELEMENTS: usize = (GRID_X * GRID_Y * GRID_Z) as usize; // {}",
            total
        )
        .unwrap();
        writeln!(
            self.code,
            "const OUTPUT_BUFFER_SIZE: u64 = (TOTAL_ELEMENTS * size_of::<f32>()) as u64;"
        )
        .unwrap();
        writeln!(
            self.code,
            "const PERM_GENERATOR_SIZE: u64 = (256 * size_of::<i32>() + 3 * size_of::<f32>()) as u64; // 1036"
        )
        .unwrap();
        writeln!(
            self.code,
            "const DENSITY_PARAMS_SIZE: u64 = 64; // DensityParams struct: origin(16) + dimensions(16) + origin_scale(16) + position_scale(16)"
        )
        .unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_struct(&mut self, name: &str, shaders: &[&ShaderDependency<'_>]) {
        writeln!(self.code, "pub struct GpuOrchestrator_{} {{", name).unwrap();
        writeln!(self.code, "    device: wgpu::Device,").unwrap();
        writeln!(self.code, "    queue: wgpu::Queue,").unwrap();
        writeln!(self.code).unwrap();

        // Pipelines
        writeln!(self.code, "    // Compute pipelines").unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "    pipeline_{}: wgpu::ComputePipeline,", sn).unwrap();
        }
        writeln!(self.code).unwrap();

        // Output buffers
        writeln!(self.code, "    // Output storage buffers (one per shader)").unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "    buf_{}_out: wgpu::Buffer,", sn).unwrap();
        }
        writeln!(self.code).unwrap();

        // Staging buffer
        writeln!(self.code, "    buf_staging: wgpu::Buffer,").unwrap();
        writeln!(self.code).unwrap();

        // Per-shader DensityParams uniform buffers (origin, dimensions, origin_scale, position_scale)
        writeln!(
            self.code,
            "    // DensityParams uniform buffers (per shader invocation)"
        )
        .unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "    buf_{sn}_params: wgpu::Buffer,").unwrap();
        }
        writeln!(self.code).unwrap();

        // Packed density input buffers (only for shaders with multiple density inputs)
        {
            let mut any = false;
            for s in shaders {
                if s.shader.inputs.len() > 1 {
                    if !any {
                        writeln!(self.code, "    // Packed density-input storage buffers").unwrap();
                        any = true;
                    }
                    let sn = shader_dep_name(s);
                    writeln!(self.code, "    buf_{sn}_density_inputs: wgpu::Buffer,").unwrap();
                }
            }
            if any {
                writeln!(self.code).unwrap();
            }
        }

        // Packed permutation table buffers (one per shader that has perm tables)
        {
            let mut any = false;
            for s in shaders {
                if !s.shader.permutation_tables.is_empty() {
                    if !any {
                        writeln!(self.code, "    // Packed permutation-table storage buffers")
                            .unwrap();
                        any = true;
                    }
                    let sn = shader_dep_name(s);
                    writeln!(self.code, "    buf_{sn}_perm_tables: wgpu::Buffer,").unwrap();
                }
            }
            if any {
                writeln!(self.code).unwrap();
            }
        }

        // Bind groups
        writeln!(self.code, "    // Pre-built bind groups").unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "    bind_group_{}: wgpu::BindGroup,", sn).unwrap();
        }

        writeln!(self.code, "}}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_impl_open(&mut self, name: &str) {
        writeln!(self.code, "impl GpuOrchestrator_{} {{", name).unwrap();
    }

    fn emit_impl_close(&mut self) {
        writeln!(self.code, "}}").unwrap();
    }

    fn emit_new(
        &mut self,
        _name: &str,
        shaders: &[&ShaderDependency<'_>],
        shader_index: &HashMap<&ShaderDependency<'_>, usize>,
        waves: &[Vec<ShaderDependency<'_>>],
    ) {
        writeln!(self.code, "    pub fn new() -> Self {{").unwrap();

        // wgpu instance / adapter / device
        writeln!(
            self.code,
            "        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());"
        )
        .unwrap();
        writeln!(self.code, "        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {{").unwrap();
        writeln!(
            self.code,
            "            power_preference: wgpu::PowerPreference::HighPerformance,"
        )
        .unwrap();
        writeln!(self.code, "            ..Default::default()").unwrap();
        writeln!(
            self.code,
            "        }})).expect(\"No suitable GPU adapter found\");"
        )
        .unwrap();
        writeln!(self.code).unwrap();
        writeln!(self.code, "        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))").unwrap();
        writeln!(
            self.code,
            "            .expect(\"Failed to request GPU device\");"
        )
        .unwrap();
        writeln!(self.code).unwrap();

        // Shader modules
        writeln!(self.code, "        // --- Load shader modules ---").unwrap();
        for s in shaders {
            let sn = sanitize_name(&s.shader.name);
            writeln!(
                self.code,
                "        let sm_{sn} = device.create_shader_module(wgpu::ShaderModuleDescriptor {{",
            )
            .unwrap();
            writeln!(self.code, "            label: Some(\"{sn}\"),").unwrap();
            writeln!(
                self.code,
                "            source: wgpu::ShaderSource::Wgsl(include_str!(\"../shaders/{sn}.wgsl\").into()),"
            )
            .unwrap();
            writeln!(self.code, "        }});").unwrap();
        }
        writeln!(self.code).unwrap();

        // Compute pipelines
        writeln!(self.code, "        // --- Create compute pipelines ---").unwrap();
        for s in shaders {
            let sn = sanitize_name(&s.shader.name);
            let dep_name = shader_dep_name(s);
            // Use entry_point: None since each shader module has exactly one
            // @compute entry point. The naga WGSL writer may mangle the name
            // (e.g. append trailing underscores), so auto-detection is safest.
            writeln!(
                self.code,
                "        let pipeline_{dep_name} = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {{",
            )
            .unwrap();
            writeln!(
                self.code,
                "            label: Some(\"pipeline_{dep_name}\"),"
            )
            .unwrap();
            writeln!(self.code, "            layout: None,").unwrap();
            writeln!(self.code, "            module: &sm_{sn},").unwrap();
            writeln!(self.code, "            entry_point: None,",).unwrap();
            writeln!(
                self.code,
                "            compilation_options: Default::default(),"
            )
            .unwrap();
            writeln!(self.code, "            cache: None,").unwrap();
            writeln!(self.code, "        }});").unwrap();
        }
        writeln!(self.code).unwrap();

        // Buffers
        writeln!(self.code, "        // --- Create buffers ---").unwrap();
        writeln!(
            self.code,
            "        let storage_out_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;"
        )
        .unwrap();
        for dep in shaders {
            let sn = shader_dep_name(dep);
            writeln!(
                self.code,
                "        let buf_{sn}_out = device.create_buffer(&wgpu::BufferDescriptor {{",
            )
            .unwrap();
            writeln!(self.code, "            label: Some(\"{sn}_out\"),").unwrap();
            // Storage buffers must match Naga's padded array layout (multiple of 4 f32).
            let buf_size =
                ensure_alignment(dep.dimensions.flatten()) as usize * std::mem::size_of::<f32>();
            writeln!(self.code, "            size: {} as u64,", buf_size).unwrap();
            writeln!(self.code, "            usage: storage_out_usage,").unwrap();
            writeln!(self.code, "            mapped_at_creation: false,").unwrap();
            writeln!(self.code, "        }});").unwrap();
        }
        writeln!(self.code).unwrap();

        // Staging
        writeln!(
            self.code,
            "        let buf_staging = device.create_buffer(&wgpu::BufferDescriptor {{"
        )
        .unwrap();
        writeln!(self.code, "            label: Some(\"staging\"),").unwrap();
        writeln!(self.code, "            size: OUTPUT_BUFFER_SIZE,").unwrap();
        writeln!(
            self.code,
            "            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,"
        )
        .unwrap();
        writeln!(self.code, "            mapped_at_creation: false,").unwrap();
        writeln!(self.code, "        }});").unwrap();
        writeln!(self.code).unwrap();

        // Uniform buffers - per-shader DensityParams
        writeln!(
            self.code,
            "        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;"
        )
        .unwrap();

        // Per-shader params buffers (containing origin, dimensions, origin_scale, position_scale)
        // Struct layout: origin(vec3f, 16 bytes), dimensions(vec3u, 16 bytes), origin_scale(vec3f, 16 bytes), position_scale(vec3f, 16 bytes)
        for s in shaders {
            let sn = shader_dep_name(s);
            let (gx, gy, gz) = s.dimensions;
            let (ox, oy, oz) = s.scaled_origin.as_float();
            let (px, py, pz) = s.scaled_position.as_float();

            writeln!(
                self.code,
                "        let buf_{sn}_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {{"
            )
            .unwrap();
            writeln!(self.code, "            label: Some(\"{sn}_params\"),").unwrap();
            // Build the DensityParams struct directly as bytes
            // origin (vec3f + padding), dimensions (vec3u + padding), origin_scale (vec3f + padding), position_scale (vec3f + padding)
            writeln!(self.code, "            contents: &{{").unwrap();
            writeln!(self.code, "                let mut params = [0u8; 64];").unwrap();
            writeln!(
                self.code,
                "                // origin at offset 0 (will be set at dispatch)"
            )
            .unwrap();
            writeln!(
                self.code,
                "                // 0..16 is origin (vec3f + pad)"
            )
            .unwrap();
            writeln!(
                self.code,
                "                // dimensions at offset 16 (vec3u + pad)"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[16..20].copy_from_slice(&{gx}_u32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[20..24].copy_from_slice(&{gy}_u32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[24..28].copy_from_slice(&{gz}_u32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(
                self.code,
                "                // origin_scale at offset 32 (vec3f + pad)"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[32..36].copy_from_slice(&{ox}_f32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[36..40].copy_from_slice(&{oy}_f32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[40..44].copy_from_slice(&{oz}_f32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(
                self.code,
                "                // position_scale at offset 48 (vec3f + pad)"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[48..52].copy_from_slice(&{px}_f32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[52..56].copy_from_slice(&{py}_f32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(
                self.code,
                "                params[56..60].copy_from_slice(&{pz}_f32.to_ne_bytes());"
            )
            .unwrap();
            writeln!(self.code, "                params").unwrap();
            writeln!(self.code, "            }},").unwrap();
            writeln!(self.code, "            usage: uniform_usage,").unwrap();
            writeln!(self.code, "        }});").unwrap();
        }
        writeln!(self.code).unwrap();

        // Packed density-input and permutation-table buffers
        writeln!(
            self.code,
            "        let packed_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;"
        )
        .unwrap();

        for s in shaders {
            if s.shader.inputs.len() > 1 {
                let sn = shader_dep_name(s);
                let total_size: u64 = s
                    .shader
                    .inputs
                    .iter()
                    .map(|dep| ensure_alignment(dep.dimensions.flatten()) as u64 * 4)
                    .sum();
                writeln!(
                    self.code,
                    "        let buf_{sn}_density_inputs = device.create_buffer(&wgpu::BufferDescriptor {{",
                )
                .unwrap();
                writeln!(
                    self.code,
                    "            label: Some(\"{sn}_density_inputs\"),"
                )
                .unwrap();
                writeln!(self.code, "            size: {},", total_size).unwrap();
                writeln!(self.code, "            usage: packed_usage,").unwrap();
                writeln!(self.code, "            mapped_at_creation: false,").unwrap();
                writeln!(self.code, "        }});").unwrap();
            }
        }

        for s in shaders {
            if !s.shader.permutation_tables.is_empty() {
                let sn = shader_dep_name(s);
                let total_size: u64 = s.shader.permutation_tables.len() as u64 * 1036;
                writeln!(
                    self.code,
                    "        let buf_{sn}_perm_tables = device.create_buffer(&wgpu::BufferDescriptor {{",
                )
                .unwrap();
                writeln!(self.code, "            label: Some(\"{sn}_perm_tables\"),").unwrap();
                writeln!(self.code, "            size: {},", total_size).unwrap();
                writeln!(self.code, "            usage: packed_usage,").unwrap();
                writeln!(self.code, "            mapped_at_creation: false,").unwrap();
                writeln!(self.code, "        }});").unwrap();
            }
        }
        writeln!(self.code).unwrap();

        // Bind groups
        writeln!(self.code, "        // --- Create bind groups ---").unwrap();
        for s in shaders {
            let sn = sanitize_name(&s.shader.name);
            let dep_name = shader_dep_name(s);
            writeln!(
                self.code,
                "        let bind_group_{dep_name} = device.create_bind_group(&wgpu::BindGroupDescriptor {{",
            )
            .unwrap();
            writeln!(self.code, "            label: Some(\"bg_{sn}\"),").unwrap();
            writeln!(
                self.code,
                "            layout: &pipeline_{dep_name}.get_bind_group_layout(0),"
            )
            .unwrap();
            writeln!(self.code, "            entries: &[").unwrap();

            // binding 0: params (DensityParams struct uniform)
            writeln!(
                self.code,
                "                buf_entry(0, &buf_{dep_name}_params, DENSITY_PARAMS_SIZE),"
            )
            .unwrap();
            // binding 1: output (storage rw)
            writeln!(
                self.code,
                "                buf_entry_whole(1, &buf_{dep_name}_out),"
            )
            .unwrap();

            let mut binding = 2u32;

            // Density inputs: bind upstream output directly for single-input shaders,
            // otherwise use the packed input buffer.
            if s.shader.inputs.len() == 1 {
                let input_sn = shader_dep_name(&s.shader.inputs[0]);
                writeln!(
                    self.code,
                    "                buf_entry_whole({binding}, &buf_{input_sn}_out),"
                )
                .unwrap();
                binding += 1;
            } else if s.shader.inputs.len() > 1 {
                writeln!(
                    self.code,
                    "                buf_entry_whole({binding}, &buf_{dep_name}_density_inputs),"
                )
                .unwrap();
                binding += 1;
            }

            // Packed permutation tables buffer
            if !s.shader.permutation_tables.is_empty() {
                writeln!(
                    self.code,
                    "                buf_entry_whole({binding}, &buf_{dep_name}_perm_tables),"
                )
                .unwrap();
                binding += 1;
            }

            writeln!(self.code, "            ],").unwrap();
            writeln!(self.code, "        }});").unwrap();
        }
        writeln!(self.code).unwrap();

        // Self constructor
        writeln!(self.code, "        Self {{").unwrap();
        writeln!(self.code, "            device,").unwrap();
        writeln!(self.code, "            queue,").unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "            pipeline_{sn},").unwrap();
        }
        for dep in shaders {
            let sn = shader_dep_name(dep);
            writeln!(self.code, "            buf_{sn}_out,").unwrap();
        }
        writeln!(self.code, "            buf_staging,").unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "            buf_{sn}_params,").unwrap();
        }
        for s in shaders {
            if s.shader.inputs.len() > 1 {
                let sn = shader_dep_name(s);
                writeln!(self.code, "            buf_{sn}_density_inputs,").unwrap();
            }
        }
        for s in shaders {
            if !s.shader.permutation_tables.is_empty() {
                let sn = shader_dep_name(s);
                writeln!(self.code, "            buf_{sn}_perm_tables,").unwrap();
            }
        }
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(self.code, "            bind_group_{sn},").unwrap();
        }
        writeln!(self.code, "        }}").unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_orchestrate(
        &mut self,
        _name: &str,
        shaders: &[&ShaderDependency<'_>],
        _shader_index: &HashMap<&ShaderDependency<'_>, usize>,
        waves: &[Vec<ShaderDependency<'_>>],
        target_idx: usize,
    ) {
        writeln!(
            self.code,
            "    /// Run the full density pipeline on the GPU and return the target output."
        )
        .unwrap();
        writeln!(
            self.code,
            "    pub fn orchestrate(&self, origin: Vec3, perm_tables: &PermutationTables) -> Vec<f32> {{"
        )
        .unwrap();

        // Upload origin to all shader params buffers (at offset 0 in DensityParams)
        writeln!(
            self.code,
            "        let origin_data: [f32; 4] = [origin.x as f32, origin.y as f32, origin.z as f32, 0.0];"
        )
        .unwrap();
        for s in shaders {
            let sn = shader_dep_name(s);
            writeln!(
                self.code,
                "        self.queue.write_buffer(&self.buf_{sn}_params, 0, bytemuck::cast_slice(&origin_data));"
            )
            .unwrap();
        }
        writeln!(self.code).unwrap();

        // Upload packed permutation tables for each shader
        for s in shaders {
            if !s.shader.permutation_tables.is_empty() {
                let sn = shader_dep_name(s);
                writeln!(self.code, "        {{").unwrap();
                writeln!(self.code, "            let mut packed = Vec::new();").unwrap();
                for pt in &s.shader.permutation_tables {
                    let field = builders::perm_table_field_name(pt);
                    writeln!(
                        self.code,
                        "            packed.extend_from_slice(&perm_generator_bytes(&perm_tables.{field}));"
                    )
                    .unwrap();
                }
                writeln!(
                    self.code,
                    "            self.queue.write_buffer(&self.buf_{sn}_perm_tables, 0, &packed);"
                )
                .unwrap();
                writeln!(self.code, "        }}").unwrap();
            }
        }
        writeln!(self.code).unwrap();

        // Command encoder
        writeln!(
            self.code,
            "        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {{"
        )
        .unwrap();
        writeln!(self.code, "            label: Some(\"density_encoder\"),").unwrap();
        writeln!(self.code, "        }});").unwrap();
        writeln!(self.code).unwrap();

        // Dispatch waves
        for (wave_idx, wave) in waves.iter().enumerate() {
            writeln!(
                self.code,
                "        // Wave {wave_idx}: {}",
                wave.iter()
                    .map(|s| sanitize_name(&s.shader.name))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .unwrap();

            // Copy dependency outputs → packed density_inputs buffers
            for dep in wave {
                if dep.shader.inputs.len() > 1 {
                    let dep_name = shader_dep_name(dep);
                    let mut offset = 0u64;
                    for input_dep in &dep.shader.inputs {
                        let input_sn = shader_dep_name(input_dep);
                        let copy_size = input_dep.dimensions.flatten() as u64
                            * std::mem::size_of::<f32>() as u64;
                        writeln!(
                            self.code,
                            "        encoder.copy_buffer_to_buffer(&self.buf_{input_sn}_out, 0, &self.buf_{dep_name}_density_inputs, {offset}, {copy_size});"
                        )
                        .unwrap();
                        offset += ensure_alignment(input_dep.dimensions.flatten()) as u64 * 4;
                    }
                }
            }

            writeln!(self.code, "        {{").unwrap();
            writeln!(
                self.code,
                "            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {{"
            )
            .unwrap();
            writeln!(
                self.code,
                "                label: Some(\"wave_{wave_idx}\"),"
            )
            .unwrap();
            writeln!(self.code, "                timestamp_writes: None,").unwrap();
            writeln!(self.code, "            }});").unwrap();

            for dep in wave {
                //let sn = sanitize_name(&dep.shader.name);
                let dep_name = shader_dep_name(dep);
                writeln!(
                    self.code,
                    "            pass.set_pipeline(&self.pipeline_{dep_name});"
                )
                .unwrap();
                writeln!(
                    self.code,
                    "            pass.set_bind_group(0, &self.bind_group_{dep_name}, &[]);"
                )
                .unwrap();
                writeln!(
                    self.code,
                    "            pass.dispatch_workgroups(4, GRID_Y / 8, 4);"
                )
                .unwrap();
            }
            writeln!(self.code, "        }}").unwrap();
            writeln!(self.code).unwrap();
        }

        // Copy target output to staging
        let target_sn = shader_dep_name(&shaders[target_idx]);
        //sanitize_name(&shaders[target_idx].shader.name);
        writeln!(
            self.code,
            "        encoder.copy_buffer_to_buffer(&self.buf_{target_sn}_out, 0, &self.buf_staging, 0, OUTPUT_BUFFER_SIZE);"
        )
        .unwrap();
        writeln!(self.code).unwrap();
        writeln!(
            self.code,
            "        self.queue.submit(std::iter::once(encoder.finish()));"
        )
        .unwrap();
        writeln!(self.code).unwrap();

        // Readback
        writeln!(
            self.code,
            "        let buffer_slice = self.buf_staging.slice(..);"
        )
        .unwrap();
        writeln!(
            self.code,
            "        let (sender, receiver) = std::sync::mpsc::channel();"
        )
        .unwrap();
        writeln!(
            self.code,
            "        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {{"
        )
        .unwrap();
        writeln!(self.code, "            sender.send(result).unwrap();").unwrap();
        writeln!(self.code, "        }});").unwrap();
        writeln!(
            self.code,
            "        self.device.poll(wgpu::PollType::Wait).unwrap();"
        )
        .unwrap();
        writeln!(
            self.code,
            "        let map_result = receiver.recv().unwrap();"
        )
        .unwrap();
        writeln!(self.code, "        map_result.unwrap();").unwrap();
        writeln!(self.code).unwrap();
        writeln!(
            self.code,
            "        let data = buffer_slice.get_mapped_range();"
        )
        .unwrap();
        writeln!(
            self.code,
            "        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();"
        )
        .unwrap();
        writeln!(self.code, "        drop(data);").unwrap();
        writeln!(self.code, "        self.buf_staging.unmap();").unwrap();
        writeln!(self.code).unwrap();
        writeln!(self.code, "        result").unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code).unwrap();
    }

    fn emit_helpers(&mut self) {
        writeln!(self.code).unwrap();
        writeln!(
            self.code,
            "fn perm_generator_bytes(sampler: &PerlinNoiseSampler) -> Vec<u8> {{"
        )
        .unwrap();
        writeln!(self.code, "    let mut bytes = Vec::with_capacity(1036);").unwrap();
        writeln!(self.code, "    for &b in sampler.permutation.iter() {{").unwrap();
        writeln!(
            self.code,
            "        bytes.extend_from_slice(&(b as i32).to_le_bytes());"
        )
        .unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(
            self.code,
            "    bytes.extend_from_slice(&(sampler.origin_x as f32).to_le_bytes());"
        )
        .unwrap();
        writeln!(
            self.code,
            "    bytes.extend_from_slice(&(sampler.origin_y as f32).to_le_bytes());"
        )
        .unwrap();
        writeln!(
            self.code,
            "    bytes.extend_from_slice(&(sampler.origin_z as f32).to_le_bytes());"
        )
        .unwrap();
        writeln!(self.code, "    bytes").unwrap();
        writeln!(self.code, "}}").unwrap();
        writeln!(self.code).unwrap();

        writeln!(
            self.code,
            "fn buf_entry(binding: u32, buffer: &wgpu::Buffer, size: u64) -> wgpu::BindGroupEntry<'_> {{"
        )
        .unwrap();
        writeln!(self.code, "    wgpu::BindGroupEntry {{").unwrap();
        writeln!(self.code, "        binding,").unwrap();
        writeln!(
            self.code,
            "        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {{"
        )
        .unwrap();
        writeln!(self.code, "            buffer,").unwrap();
        writeln!(self.code, "            offset: 0,").unwrap();
        writeln!(
            self.code,
            "            size: Some(std::num::NonZeroU64::new(size).unwrap()),"
        )
        .unwrap();
        writeln!(self.code, "        }}),").unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code, "}}").unwrap();
        writeln!(self.code).unwrap();

        writeln!(
            self.code,
            "fn buf_entry_whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {{"
        )
        .unwrap();
        writeln!(self.code, "    wgpu::BindGroupEntry {{").unwrap();
        writeln!(self.code, "        binding,").unwrap();
        writeln!(self.code, "        resource: buffer.as_entire_binding(),").unwrap();
        writeln!(self.code, "    }}").unwrap();
        writeln!(self.code, "}}").unwrap();
    }
}

fn shader_dep_name(dep: &ShaderDependency<'_>) -> String {
    sanitize_name(&format!("{}_{}", dep.shader.name, dep.dimensions.flatten()))
}

/// Pad an element count to the next multiple of 4 (matching the Naga transform).
fn ensure_alignment(len: usize) -> u32 {
    let l = len as u32;
    if l % 4 == 0 { l } else { l + (4 - (l % 4)) }
}
