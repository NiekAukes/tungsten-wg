use std::{collections::HashMap, env};

use serde::de;

use crate::{
    rcl::codegen::RustCodeGenerator,
    spmt::{
        dag::DensityDAG,
        model::{Addr, DensityFunctionRef},
        pretty::PrettyPrint,
    },
    transform_orchestration_rcl::OrchestrationConverter,
};

pub mod config_load;
pub mod parse;
pub mod spmt;
//pub mod tape;
pub mod tungsten_parse;

pub mod orchestrate;

pub mod transform_spmt;

pub mod rcl;
pub mod transform_orchestration_rcl;
pub mod transform_rcl;

pub fn main() {
    let mut data = config_load::MinecraftDataRaw::new();
    config_load::load_all_configs(&mut data, "vanilla_worldgen_1.21.1", None);
    //config_load::load_all_configs(&mut data, "JJThunderToTheMax", None);
    config_load::load_all_configs(&mut data, "testmod", None);
    // reexport the config
    // config_load::reexport(&data, "reexport_t");

    let arena: bumpalo::Bump = bumpalo::Bump::with_capacity(1 * 1024 * 1024); // 1 MB initial capacity
    let mut mcdata = parse::MinecraftData::new(&arena, &data);
    mcdata.parse_from_raw();
    println!(
        "Parsed Minecraft data: {} density functions",
        mcdata.density_functions.len()
    );
    println!(
        "Final arena usage after parsing: {} MB",
        arena.allocated_bytes() as f64 / (1024.0 * 1024.0)
    );

    let transform_arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024); // 1 MB initial capacity
    let transformer = transform_spmt::Transformer::new(&transform_arena);
    //println!("noise seetings keys: {:?}", mcdata.noise_settings.keys());
    let noise_generator = mcdata.noise_settings.get("minecraft:overworld").unwrap();
    //println!("Transforming noise generator: {:?}", noise_generator);
    let program = transformer.transform(noise_generator);

    drop(arena);

    // print the final arena usage
    let bytes = transform_arena.allocated_bytes();
    println!(
        "Final arena usage after SPMT transformation: {} MB",
        bytes as f64 / (1024.0 * 1024.0)
    );

    let mut printer = spmt::pretty::Printer::new();
    program.pretty(&mut printer);

    let (out, name_cache) = printer.finish_with_name_cache();
    // write the output to a file
    std::fs::write("output.spmt", out).expect("Unable to write file");

    // create a folder for the density DAGs

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

    // transform to orchestration
    let orchestration_arena = bumpalo::Bump::new();
    let orchestration = orchestrate::transform::transform_from_spmt(&program, &orchestration_arena);

    let mut printer = spmt::pretty::Printer::new();
    orchestration.pretty_wave_graph(&mut printer);
    let orchestration_output = printer.finish();
    std::fs::write("wave_graph.dot", orchestration_output).expect("Unable to write file");

    let mut printer = spmt::pretty::Printer::new();
    orchestration.pretty_wave_dependencies(&mut printer);
    let orchestration_output = printer.finish();
    std::fs::write("wave_dependencies.dot", orchestration_output).expect("Unable to write file");

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
    let folder = "../rcl_density";
    // generate the

    let rust_cg = RustCodeGenerator::new();
    //println!("RCL Model: {:?}", rcl_model);
    let rcl_output = rust_cg.generate_module(&rcl_model);
    let orch_output = rust_cg.generate_module(&orchestration_rcl);
    std::fs::write(format!("{}/src/density_function.rs", folder), rcl_output)
        .expect("Unable to write file");
    std::fs::write(format!("{}/src/orchestration.rs", folder), orch_output)
        .expect("Unable to write file");
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
        let mut mcdata = parse::MinecraftData::new(&arena, &data);
        mcdata.parse_from_raw();
        println!("{:?}", mcdata);
    }
}
