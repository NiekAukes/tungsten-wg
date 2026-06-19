#![allow(warnings)]
use crate::{rcl::codegen::RustCodeGenerator, transform_orchestration_rcl::OrchestrationConverter};

pub mod config_load;
pub mod parse;
pub mod spmt;
//pub mod tape;
pub mod tungsten_parse;

pub mod orchestrate;

pub mod transform_spmt;

pub mod cuda;
pub mod rcl;
pub mod transform_cuda;
pub mod transform_naga;
pub mod transform_orchestration_cuda;
pub mod transform_orchestration_gpu;
pub mod transform_orchestration_rcl;
pub mod transform_rcl;

pub fn run_inline_generation(path: &str, base: &str) -> (String, String) {
    let mut data = config_load::MinecraftDataRaw::new();
    config_load::load_all_configs(&mut data, base, None);
    config_load::load_all_configs(&mut data, path, None);
    let arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024);
    let mut mcdata = parse::MinecraftData::new(&arena, &data, 16);
    mcdata.parse_from_raw();

    let noise_generator = mcdata
        .noise_settings
        .get("minecraft:overworld")
        .expect("Could not find minecraft:overworld settings");

    let transform_arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024);
    let transformer = transform_spmt::Transformer::new(&transform_arena);
    let program = transformer.transform(noise_generator);

    // run orchestration conversion and codegen
    let orchestration_arena = bumpalo::Bump::new();
    let orchestration = orchestrate::transform::transform_from_spmt(&program, &orchestration_arena);
    let mut orchestration_conv = OrchestrationConverter::new(&orchestration_arena);
    let waves = orchestration.arrange_waves();
    orchestration_conv.convert(&waves, orchestration.get_primary_shaders());
    let orchestration_rcl = orchestration_conv.finish();
    let orch_output = RustCodeGenerator.generate_inline_module(&orchestration_rcl, "orchestration");

    // generate Rust code for the main density functions
    let rcl_model =
        transform_rcl::convert_spmt_to_inline_rcl(&program, &waves, &orchestration_arena);
    let rcl_output = RustCodeGenerator.generate_inline_module(&rcl_model, "density_function");

    (rcl_output, orch_output)
}
