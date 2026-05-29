use std::{collections::HashMap, env, path::PathBuf, thread::Builder};

use clap::{Parser, arg, command};
use serde::de;

use crate::{
    rcl::codegen::RustCodeGenerator,
    spmt::{
        dag::DensityDAG,
        model::{Addr, DensityFunctionRef},
        pretty::PrettyPrint,
    },
    transform_orchestration_gpu::GpuOrchestrationCodegen,
    transform_orchestration_rcl::OrchestrationConverter,
};

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

#[derive(Parser, Debug)]
#[command(author, version, about = "Minecraft Worldgen SPMT Transformer & Codegen", long_about = None)]
struct Args {
    /// The mod folder to process (e.g., "JJThunderToTheMax")
    #[arg(short, long, default_value = "vanilla_worldgen_1.21.1")]
    mod_folder: String,

    /// The base Minecraft worldgen version/folder to load
    #[arg(short, long, default_value = "vanilla_worldgen_1.21.1")]
    base_version: String,

    /// Output directory for generated code and shaders
    #[arg(short, long, default_value = "../rcl_density")]
    output_dir: PathBuf,

    /// Whether to skip WGSL shader generation
    #[arg(long, default_value_t = false)]
    skip_shaders: bool,

    // chunks per batch
    /// Chunk size for generation (e.g., 16 for 16x16 chunks, 32 for 32x32, etc.)
    /// Only useful for benchmarking for now
    /// Maximum is 128 for cross compatibility on GPUs
    #[arg(long, default_value_t = 16)]
    chunk_size: usize,

    /// Whether to output intermediate representations to files
    #[arg(long, default_value_t = false)]
    emit_intermediates: bool,

    /// Enable verbose logging of arena usage
    #[arg(short, long)]
    verbose: bool,
}

pub fn main() {
    let args = Args::parse();

    let h = Builder::new()
        .name("Main Thread".into())
        .stack_size(16 * 1024 * 1024) // 16 MB stack size
        .spawn(move || {
            run_with_args(args);
        })
        .expect("Failed to spawn main thread");
    h.join().expect("Main thread panicked");
}

fn run_with_args(args: Args) {
    let mut data = config_load::MinecraftDataRaw::new();

    // Load configs based on CLI args
    config_load::load_all_configs(&mut data, &args.base_version, None);
    if args.mod_folder != args.base_version {
        config_load::load_all_configs(&mut data, &args.mod_folder, None);
    }

    let arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024);
    let mut mcdata = parse::MinecraftData::new(&arena, &data, args.chunk_size);
    mcdata.parse_from_raw();

    if args.verbose {
        println!(
            "Final arena usage after parsing: {:.2} MB",
            arena.allocated_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    // --- Core Logic (Density Functions & Transformations) ---
    let noise_generator = mcdata
        .noise_settings
        .get("minecraft:overworld")
        .expect("Could not find minecraft:overworld settings");

    let transform_arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024);
    let transformer = transform_spmt::Transformer::new(&transform_arena);
    let program = transformer.transform(noise_generator);

    if args.verbose {
        let bytes = transform_arena.allocated_bytes();
        println!(
            "Final arena usage after SPMT transformation: {:.2} MB",
            bytes as f64 / (1024.0 * 1024.0)
        );
    }

    if args.emit_intermediates {
        // pretty print one of the density functions to a file
        let density_function = mcdata
            .noise_settings
            .get("minecraft:overworld")
            .unwrap()
            .noise_router
            .final_density;
        let pretty = format!("{}", density_function.get_density());
        std::fs::write("pretty_density_function.txt", pretty).expect("Unable to write file");
        let dot = parse::dot::print_density_dot(density_function.get_density());

        std::fs::write("density_function.dot", dot).expect("Unable to write file");
    }

    let transform_arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024); // 1 MB initial capacity
    let transformer = transform_spmt::Transformer::new(&transform_arena);
    //println!("noise seetings keys: {:?}", mcdata.noise_settings.keys());
    let noise_generator = mcdata.noise_settings.get("minecraft:overworld").unwrap();
    //println!("Transforming noise generator: {:?}", noise_generator);
    let program = transformer.transform(noise_generator);

    drop(arena);

    if args.verbose {
        // print the final arena usage
        let bytes = transform_arena.allocated_bytes();
        println!(
            "Final arena usage after SPMT transformation: {} MB",
            bytes as f64 / (1024.0 * 1024.0)
        );
    }

    // create a folder for the density DAGs

    if args.emit_intermediates {
        let mut printer = spmt::pretty::Printer::new();
        program.pretty(&mut printer);
        let (_, name_cache) = printer.finish_with_name_cache();
        std::fs::create_dir_all("density_dags").expect("Unable to create directory");
        let mut i = 0;
        let mut name_cache_bor = Some(name_cache);
        for (density_function, _) in &program.main_density_functions {
            let ddag_root = *density_function;
            let ddag = DensityDAG { root: ddag_root };
            let mut printer =
                spmt::pretty::Printer::new_with_name_cache(name_cache_bor.take().unwrap());
            ddag.pretty(&mut printer);
            let (dot_output, name_cache) = printer.finish_with_name_cache();

            // sanitize the file name by replacing any characters that are not alphanumeric or underscores with underscores
            let fname = density_function.canonical_name.clone().unwrap_or_else(|| {
                name_cache
                    .get(&(*density_function).addr())
                    .cloned()
                    .unwrap_or_else(|| "unknown".into())
            });

            name_cache_bor = Some(name_cache);
            let file_name = format!(
                "density_dags/{}.dot",
                fname.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
            );

            match std::fs::write(&file_name, dot_output) {
                Ok(_) => (),
                Err(e) => {
                    eprintln!("Failed to write DOT file for density function {}: {}", i, e);
                    eprintln!("file name: {}", &file_name);
                }
            }
            i += 1;
        }

        std::fs::create_dir_all("density_functions").expect("Unable to create directory");
        let mut i = 0;

        for density_function in &program.density_functions {
            // pretty print the density function to a file
            let mut printer =
                spmt::pretty::Printer::new_with_name_cache(name_cache_bor.take().unwrap());
            density_function.pretty_with_deps(&mut printer);
            let (dot_output, name_cache) = printer.finish_with_name_cache();

            let fname = density_function.canonical_name.clone().unwrap_or_else(|| {
                name_cache
                    .get(&(*density_function).addr())
                    .cloned()
                    .unwrap_or_else(|| "unknown".into())
            });
            name_cache_bor = Some(name_cache);
            let file_name = format!(
                "density_functions/{}.spmt",
                fname.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
            );

            match std::fs::write(&file_name, dot_output) {
                Ok(_) => (),
                Err(e) => {
                    eprintln!(
                        "Failed to write SPMT file for density function {}: {}",
                        i, e
                    );
                    eprintln!("file name: {}", &file_name);
                }
            }
            i += 1;
        }
    }

    // transform to orchestration
    let orchestration_arena = bumpalo::Bump::new();
    let orchestration = orchestrate::transform::transform_from_spmt(&program, &orchestration_arena);

    if args.emit_intermediates {
        let mut printer = spmt::pretty::Printer::new();
        orchestration.pretty_wave_graph(&mut printer);
        let orchestration_output = printer.finish();
        std::fs::write("wave_graph.dot", orchestration_output).expect("Unable to write file");

        let mut printer = spmt::pretty::Printer::new();
        orchestration.pretty_wave_dependencies(&mut printer);
        let orchestration_output = printer.finish();
        std::fs::write("wave_dependencies.dot", orchestration_output)
            .expect("Unable to write file");
    }
    // convert one of the density functions to RCL
    // let rcl_arena = bumpalo::Bump::new();
    // let mut rcl_model = rcl::RCL::new();
    // let density_function = program.density_functions.first().unwrap();
    // transform_rcl::add_density_to_rcl_model(&mut rcl_model, density_function, &rcl_arena);

    // convert all density functions to RCL and add them to the model
    let rcl_arena = bumpalo::Bump::new();
    let mut rcl_model = rcl::RCL::new();
    let mut already_converted_functions = HashMap::new();
    for density_function in &program.density_functions {
        let c = transform_rcl::add_density_to_rcl_model(
            &mut rcl_model,
            density_function,
            &rcl_arena,
            already_converted_functions,
        );
        already_converted_functions = c.already_converted_functions;
    }

    let mut orchestration_conv = OrchestrationConverter::new(&rcl_arena);
    orchestration_conv.convert(
        orchestration.arrange_waves(),
        orchestration.get_primary_shaders(),
    );

    // Generate a pruned orchestration function for each primary density
    for primary in &orchestration.get_primary_shaders() {
        let name = &primary.shader.name;
        let pruned_waves = orchestration.arrange_waves_for(primary);

        orchestration_conv.convert_single_entry(name, pruned_waves, primary);
    }

    let orchestration_rcl = orchestration_conv.finish();

    // write the RCL functions to a file in a different folder
    //let folder = "../rcl_density";
    //let folder = "../Tungsten/libtungsten";
    let folder = args.output_dir.as_path().to_str().unwrap();
    std::fs::create_dir_all(folder).expect("Unable to create output directory");

    let rust_cg = RustCodeGenerator::new();
    let rcl_output = rust_cg.generate_module(&rcl_model);
    let orch_output = rust_cg.generate_module(&orchestration_rcl);

    println!(
        "Generated 'density_function.rs' ({} bytes)",
        rcl_output.len()
    );
    println!("Generated 'orchestration.rs' ({} bytes)", orch_output.len());

    std::fs::write(format!("{}/src/density_function.rs", folder), rcl_output)
        .expect("Unable to write file");
    std::fs::write(format!("{}/src/orchestration.rs", folder), orch_output)
        .expect("Unable to write file");

    if !args.skip_shaders {
        gpu_codegen(&orchestration, &program, folder);
    }
}

fn gpu_codegen(
    orchestration: &orchestrate::model::Orchestration,
    program: &spmt::model::SPMT,
    folder: &str,
) {
    // Generate GPU orchestrator code for each primary density
    let mut gpu_codegen = GpuOrchestrationCodegen::new();
    for primary in &orchestration.get_primary_shaders() {
        let name = &primary.shader.name;
        let pruned_waves = orchestration.arrange_waves_for(primary);
        gpu_codegen.convert_single_entry(name, &pruned_waves, primary);
    }
    let gpu_orch_output = gpu_codegen.finish();
    std::fs::write(
        format!("{}/src/gpu_orchestrator.rs", folder),
        gpu_orch_output,
    )
    .expect("Unable to write GPU orchestrator file");

    // transform to Naga IR and write WGSL shaders (one per density function)
    let helpers = transform_naga::parse_helpers();
    let naga_modules =
        transform_naga::convert_spmt_to_naga(&program, transform_naga::Precision::F32, helpers);

    // delete and recreate the shaders folder

    // delete if it exists
    if std::path::Path::new(&format!("{}/shaders", folder)).exists() {
        std::fs::remove_dir_all(&format!("{}/shaders", folder))
            .expect("Unable to delete shaders directory");
    }
    std::fs::create_dir_all(&format!("{}/shaders", folder))
        .expect("Unable to create shaders directory");

    for (name, naga_module) in &naga_modules {
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::empty(),
            naga::valid::Capabilities::all(),
        );

        let naga_module = naga_module.replace(naga::Module::default());

        match validator.validate(&naga_module) {
            Ok(info) => {
                let mut wgsl_writer = naga::back::wgsl::Writer::new(
                    String::new(),
                    naga::back::wgsl::WriterFlags::empty(),
                );
                match wgsl_writer.write(&naga_module, &info) {
                    Ok(()) => {
                        let wgsl = wgsl_writer.finish();
                        let file_path = format!("{}/shaders/{}.wgsl", folder, name);
                        std::fs::write(&file_path, &wgsl).expect("Unable to write WGSL file");
                        println!(
                            "Generated WGSL shader '{}' ({} bytes)",
                            file_path,
                            wgsl.len()
                        );
                    }
                    Err(e) => {
                        eprintln!("Failed to write WGSL for '{}': {}", name, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Naga validation failed for '{}': {:?}", name, e);
            }
        }
    }

    // Generate CUDA code
    let cuda_folder = "../cuda_density";
    let cuda_arena = bumpalo::Bump::new();
    let mut cuda_module = cuda::model::CudaModule::new();
    cuda_module.add_include("\"helpers.cu\"".to_string());
    for density_function in &program.density_functions {
        transform_cuda::add_density_to_cuda_module(
            &mut cuda_module,
            density_function,
            &cuda_arena,
            HashMap::new(),
        );
    }
    let cuda_generator = cuda::codegen::CudaCodeGenerator::new();
    let cuda_output = cuda_generator.generate_module(&cuda_module);
    std::fs::write(
        format!("{}/src/density_function.cu", cuda_folder),
        cuda_output,
    )
    .expect("Unable to write CUDA file");

    let mut cuda_orchestration_codegen =
        transform_orchestration_cuda::CudaOrchestrationCodegen::new();
    let waves = orchestration.arrange_waves();
    for primary in &orchestration.get_primary_shaders() {
        let pruned_waves = orchestration.arrange_waves_for(primary);
        cuda_orchestration_codegen.convert_single_entry(
            &primary.shader.name,
            pruned_waves.as_ref(),
            primary,
        );
    }
    let cuda_orch_output = cuda_orchestration_codegen.finish();
    std::fs::write(
        format!("{}/src/orchestration.cu", cuda_folder),
        cuda_orch_output,
    )
    .expect("Unable to write CUDA orchestration file");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_load() {
        let mut data = config_load::MinecraftDataRaw::new();
        config_load::load_all_configs(&mut data, "vanilla_worldgen_1.21.1", None);
        // print length of each hashmap
        println!("Noise Settings: {}", data.noise_settings.len());
        println!("Density Functions: {}", data.density_functions.len());
        println!("Normal Noises: {}", data.normal_noises.len());

        // print all keys of each hashmap
        for key in data.noise_settings.keys() {
            println!("Noise Setting: {}", key);
        }
        for key in data.density_functions.keys() {
            println!("Density Function: {}", key);
        }
    }

    #[test]
    fn test_parse_config() {
        let mut data = config_load::MinecraftDataRaw::new();
        config_load::load_all_configs(&mut data, "vanilla_worldgen_1.19", None);
        let arena = bumpalo::Bump::with_capacity(10 * 1024 * 1024); // 10 MB initial capacity
        let mut mcdata = parse::MinecraftData::new(&arena, &data, 16);
        mcdata.parse_from_raw();
        println!("{:?}", mcdata);
    }
}
