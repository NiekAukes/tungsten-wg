use kronos_compute::api::{BufferBinding, ComputeContext, PipelineConfig, pipeline};

const SIZE_X: u32 = 256;
const SIZE_Y: u32 = 256;
const SIZE_Z: u32 = 1;
const SIZE: u32 = SIZE_X * SIZE_Y * SIZE_Z * std::mem::size_of::<f32>() as u32;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    origin: [f32; 3],
    inscale: [f32; 3],
    resolution: [u32; 3],
    //_padding2: u32,
}

pub fn doit(spv: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let ctx = ComputeContext::builder()
        .app_name("Minecraft GPU Evaluator")
        .enable_validation()
        .prefer_vendor("AMD")
        .prefer_icd_index(3)
        .build()?;

    let device_name = ctx.device_properties().deviceName;
    // convert to string
    let device_name = unsafe {
        std::ffi::CStr::from_ptr(device_name.as_ptr())
            .to_str()
            .unwrap()
    };
    println!("Using device: {}", device_name);
    // Load shader and create pipeline
    //let shader = ctx.load_shader("src/shaders/minecraft_amplified_barrier.spv")?;
    let shader = ctx.create_shader_from_spirv(spv)?;

    let pipeline_config = PipelineConfig {
        entry_point: "main".to_string(),
        local_size: (8, 8, 1),
        bindings: vec![BufferBinding {
            binding: 0,
            ..Default::default()
        }],
        push_constant_size: std::mem::size_of::<Params>() as u32,
    };

    let pipeline = ctx.create_pipeline_with_config(&shader, pipeline_config)?;

    // Create buffers
    //let input = ctx.create_buffer(&data)?;
    let output = ctx.create_buffer_uninit(SIZE as usize)?;

    let params = Params {
        origin: [0.0, 0.0, 0.0],
        inscale: [1.0, 1.0, 1.0],
        //outscale: 32.0,
        resolution: [SIZE_X, SIZE_Y, SIZE_Z],
    };

    // Dispatch compute work
    ctx.dispatch(&pipeline)
        .bind_buffer(0, &output)
        .push_constants(&params)
        .workgroups(SIZE_X / 8, SIZE_Y / 8, SIZE_Z)
        .execute()?;
    // Read results
    // wait 100ms for the GPU to finish
    let results: Vec<f32> = output.read()?;
    Ok(results)
}
