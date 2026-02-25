use std::env;

use serde::de;

use crate::spmt::{
    dag::DensityDAG,
    model::{Addr, DensityFunctionRef},
    pretty::PrettyPrint,
};

pub mod config_load;
pub mod eval_reference;
pub mod gpu_eval;
pub mod parse;
pub mod spmt;
//pub mod tape;
pub mod tungsten_parse;

pub mod orchestrate;

pub mod transform_spmt;

pub fn main() {
    let mut data = config_load::MinecraftDataRaw::new();
    config_load::load_all_configs(&mut data, "vanilla_worldgen_1.21.1", None);
    config_load::load_all_configs(&mut data, "JJThunderToTheMax", None);
    // reexport the config
    //config_load::reexport(&data, "reexport_jj");

    let arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024); // 1 MB initial capacity
    let mut mcdata = parse::MinecraftData::new(&arena, &data);
    mcdata.parse_from_raw();
    //println!("Parsed Minecraft data: {:?}", mcdata);
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
    let program = transformer.transform(noise_generator);

    drop(arena);

    // print the final arena usage
    let bytes = transform_arena.allocated_bytes();
    println!("Final arena usage: {} MB", bytes as f64 / (1024.0 * 1024.0));

    let mut printer = spmt::pretty::Printer::new();
    program.pretty(&mut printer);

    let (out, name_cache) = printer.finish_with_name_cache();
    // write the output to a file
    std::fs::write("output.spmt", out).expect("Unable to write file");

    // create a folder for the density DAGs

    std::fs::create_dir_all("density_dags").expect("Unable to create directory");
    let mut i = 0;
    let mut name_cache_bor = Some(name_cache);
    for density_function in &program.main_density_functions {
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
    let orchestration = orchestrate::transform::transform_from_spmt(program, &orchestration_arena);

    let mut printer = spmt::pretty::Printer::new();
    orchestration.pretty_wave_graph(&mut printer);
    let orchestration_output = printer.finish();
    std::fs::write("wave_graph.dot", orchestration_output).expect("Unable to write file");

    let mut printer = spmt::pretty::Printer::new();
    orchestration.pretty_wave_dependencies(&mut printer);
    let orchestration_output = printer.finish();
    std::fs::write("wave_dependencies.dot", orchestration_output).expect("Unable to write file");

    //println!("Transformed SPMT: {}", printer.finish());

    //let spv = SHADER_SOURCE.to_vec();
    //env_logger::init();

    // println!("SPIR-V length: {}", SHADER_SOURCE.len());
    // let results = gpu_eval::doit(SHADER_SOURCE).unwrap();
    // println!("Results length: {}", results.len());
    // println!("First 10 results: {:?}", &results[..100]);

    // let tung = "src/tungsten_parse/test.w";
    // let parsed = tungsten_parse::parse_tungsten(tung).unwrap();
    // println!("Parsed Tungsten file: {:?}", parsed);
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
