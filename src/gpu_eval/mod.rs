use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    origin: [f32; 3],
    _pad0: f32, // WGSL std140 alignment
    scale: [f32; 3],
    vscale: f32,
    resolution: [i32; 3],
    _pad1: i32,
}

const SIZE_X: u32 = 64;
const SIZE_Y: u32 = 64;
const SIZE_Z: u32 = 1;
const WORKGROUP_SIZE: (u32, u32, u32) = (8, 8, 1);
const OUTPUT_LEN: usize = (SIZE_X * SIZE_Y * SIZE_Z) as usize;

pub fn doit(source: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // === Initialize GPU synchronously ===
    let instance = wgpu::Instance::default();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapters found.");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Compute Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("Failed to create device.");

    let device_name = adapter.get_info().name;
    println!("Using device: {}", device_name);

    // === Load SPIR-V shader ===
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    });

    println!("Shader module created.");
    // === Buffers ===
    let output_size =
        (SIZE_X * SIZE_Y * SIZE_Z * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;

    // Output storage buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Uniform buffer for parameters
    let params = Params {
        origin: [0.0, 0.0, 0.0],
        _pad0: 0.0,
        scale: [1.0, 1.0, 1.0],
        vscale: 1.0,
        resolution: [SIZE_X as i32, SIZE_Y as i32, SIZE_Z as i32],
        _pad1: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // === Bind groups ===
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("BindGroupLayout"),
        entries: &[
            // Binding 0 → Storage buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1 → Uniform buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BindGroup"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create a "staging buffer" for readback (CPU-readable)
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // === Pipeline ===
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PipelineLayout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ComputePipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "compute_main",
        compilation_options: Default::default(),
    });

    // // === Encode dispatch ===
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ComputeEncoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ComputePass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(SIZE_X / 1, SIZE_Y / 1, SIZE_Z);
    }

    // Copy GPU → staging buffer for CPU readback
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    // === Read results ===
    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = buffer_slice.get_mapped_range();
    let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    println!(
        "Read {} results. Example: {:?}",
        results.len(),
        &results[..8.min(results.len())]
    );
    Ok(results)
}
