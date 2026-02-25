use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{collections::HashMap, fmt::Debug, fs};

/*
📂 Folder → Registry mapping

worldgen/noise_settings → NoiseGeneratorSettings registry

worldgen/density_function → DensityFunction registry

worldgen/noise → NormalNoise.NoiseParameters registry

worldgen/configured_carver → ConfiguredCarver registry

worldgen/configured_feature → ConfiguredFeature registry

worldgen/placed_feature → PlacedFeature registry

worldgen/biome → Biome registry

worldgen/structure → StructureType (or in newer versions, configured structures live under configured_structure_feature)

worldgen/structure_set → StructureSet registry

worldgen/template_pool → StructureTemplatePool (for jigsaw structures like villages)

worldgen/processor_list → StructureProcessorList (for structure block modifications)
*/
#[derive(Debug, Clone, Copy)]
pub enum MinecraftResourceType {
    NoiseGeneratorSettings,
    DensityFunction,
    NormalNoise,
    ConfiguredCarver,
    ConfiguredFeature,
    PlacedFeature,
    Biome,
    StructureType,
    StructureSet,
    StructureTemplatePool,
    StructureProcessorList,
}

pub struct MinecraftDataRaw {
    pub(crate) noise_settings: HashMap<String, String>,
    pub(crate) density_functions: HashMap<String, String>,
    pub(crate) normal_noises: HashMap<String, String>,
    configured_carvers: HashMap<String, String>,
    configured_features: HashMap<String, String>,
    placed_features: HashMap<String, String>,
    biomes: HashMap<String, String>,
    structure_types: HashMap<String, String>,
    structure_sets: HashMap<String, String>,
    structure_template_pools: HashMap<String, String>,
    structure_processor_lists: HashMap<String, String>,
}

impl MinecraftDataRaw {
    pub fn new() -> Self {
        MinecraftDataRaw {
            noise_settings: HashMap::new(),
            density_functions: HashMap::new(),
            normal_noises: HashMap::new(),
            configured_carvers: HashMap::new(),
            configured_features: HashMap::new(),
            placed_features: HashMap::new(),
            biomes: HashMap::new(),
            structure_types: HashMap::new(),
            structure_sets: HashMap::new(),
            structure_template_pools: HashMap::new(),
            structure_processor_lists: HashMap::new(),
        }
    }

    fn add_config(&mut self, name: String, config: String, resource_type: MinecraftResourceType) {
        match resource_type {
            MinecraftResourceType::NoiseGeneratorSettings => {
                self.noise_settings.insert(name, config);
            }
            MinecraftResourceType::DensityFunction => {
                self.density_functions.insert(name, config);
            }
            MinecraftResourceType::NormalNoise => {
                self.normal_noises.insert(name, config);
            }
            MinecraftResourceType::ConfiguredCarver => {
                self.configured_carvers.insert(name, config);
            }
            MinecraftResourceType::ConfiguredFeature => {
                self.configured_features.insert(name, config);
            }
            MinecraftResourceType::PlacedFeature => {
                self.placed_features.insert(name, config);
            }
            MinecraftResourceType::Biome => {
                self.biomes.insert(name, config);
            }
            MinecraftResourceType::StructureType => {
                self.structure_types.insert(name, config);
            }
            MinecraftResourceType::StructureSet => {
                self.structure_sets.insert(name, config);
            }
            MinecraftResourceType::StructureTemplatePool => {
                self.structure_template_pools.insert(name, config);
            }
            MinecraftResourceType::StructureProcessorList => {
                self.structure_processor_lists.insert(name, config);
            }
        }
    }

    fn add_config_from_file(
        &mut self,
        file_path: &std::path::Path,
        base_path: &std::path::Path,
        resource_type: MinecraftResourceType,
    ) {
        let config = std::fs::read_to_string(file_path).unwrap();

        let relative_path = file_path.strip_prefix(base_path).unwrap();
        let path_str = relative_path.to_str().unwrap();
        let relative_path = path_str.strip_suffix(".json").unwrap();
        // Prefix the name with "minecraft:name_of_file" without the leading path or file extension

        let name = format!("minecraft:{}", relative_path);
        self.add_config(name, config, resource_type);
    }
}

impl Debug for MinecraftDataRaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinecraftData")
            .field("noise_settings", &self.noise_settings.keys())
            .field("density_functions", &self.density_functions.keys())
            .field("normal_noises", &self.normal_noises.keys())
            .field("configured_carvers", &self.configured_carvers.keys())
            .field("configured_features", &self.configured_features.keys())
            .field("placed_features", &self.placed_features.keys())
            .field("biomes", &self.biomes.keys())
            .field("structure_types", &self.structure_types.keys())
            .field("structure_sets", &self.structure_sets.keys())
            .field(
                "structure_template_pools",
                &self.structure_template_pools.keys(),
            )
            .field(
                "structure_processor_lists",
                &self.structure_processor_lists.keys(),
            )
            .finish()
            .unwrap();
        Ok(())
    }
}

fn infer_resource_type_from_folder(folder_name: &str) -> Option<MinecraftResourceType> {
    match folder_name {
        "noise_settings" => Some(MinecraftResourceType::NoiseGeneratorSettings),
        "density_function" => Some(MinecraftResourceType::DensityFunction),
        "noise" => Some(MinecraftResourceType::NormalNoise),
        "configured_carver" => Some(MinecraftResourceType::ConfiguredCarver),
        "configured_feature" => Some(MinecraftResourceType::ConfiguredFeature),
        "placed_feature" => Some(MinecraftResourceType::PlacedFeature),
        "biome" => Some(MinecraftResourceType::Biome),
        "structure" => Some(MinecraftResourceType::StructureType),
        "structure_set" => Some(MinecraftResourceType::StructureSet),
        "template_pool" => Some(MinecraftResourceType::StructureTemplatePool),
        "processor_list" => Some(MinecraftResourceType::StructureProcessorList),
        _ => None,
    }
}

pub fn load_all_configs(
    minecraft_data: &mut MinecraftDataRaw,
    folder_path: &str,
    resource_type: Option<(MinecraftResourceType, &str)>,
) {
    let paths = std::fs::read_dir(folder_path).unwrap();

    for path in paths {
        let path = path.unwrap().path();

        if path.is_file() && resource_type.is_some() {
            // calculate the path of the resource relative to the base path
            let base_path = resource_type.unwrap().1;
            let path = std::path::Path::new(&path);
            let base_path = std::path::Path::new(base_path);
            minecraft_data.add_config_from_file(path, base_path, resource_type.unwrap().0);
        } else if path.is_dir() {
            // try to infer resource type from folder name
            let resource_type = if resource_type.is_none() {
                let folder_name = path.file_name().unwrap().to_str().unwrap();
                let inferred_type = infer_resource_type_from_folder(folder_name);
                inferred_type.map(|t| (t, path.to_str().unwrap()))
            } else {
                resource_type
            };
            load_all_configs(minecraft_data, path.to_str().unwrap(), resource_type);
        }
    }
}

pub fn reexport(data: &MinecraftDataRaw, out_dir: &str) {
    fs::create_dir_all(out_dir).unwrap();

    // helper macro to reduce repetition
    macro_rules! export_category {
        ($map:expr, $folder:expr, $ext:expr) => {{
            let category_dir = Path::new(out_dir).join($folder);
            fs::create_dir_all(&category_dir).unwrap();
            for (name, config) in $map {
                // turn "minecraft:overworld" → "minecraft_overworld.json"
                let safe_name = name.replace(':', "_").replace('/', "___");
                let filename = format!("{}.{}", safe_name, $ext);
                let filepath = Path::new(&category_dir).join(filename);
                fs::File::create(&filepath).unwrap();
                let content = format!("\"{}\"\n{}", name, config);
                fs::write(filepath, content).unwrap();
            }
        }};
    }
    // let category_dir = Path::new(out_dir).join("noise_settings");
    // fs::create_dir_all(&category_dir).unwrap();
    // for (name, config) in &data.noise_settings {
    //     // turn "minecraft:overworld" → "minecraft_overworld.json"
    //     let safe_name = name.replace(':', "_").replace('/', "___");
    //     let filename = format!("{}.{}", safe_name, "w");
    //     let filepath = Path::new(&category_dir).join(filename);
    //     fs::File::create(&filepath).unwrap();
    //     let content = format!("\"{}\"\n{}", name, config);
    //     fs::write(filepath, content).unwrap();
    // }

    export_category!(&data.noise_settings, "noise_settings", "w");
    export_category!(&data.density_functions, "density_functions", "density");
    export_category!(&data.normal_noises, "normal_noises", "noise");
    // export_category!(
    //     data.configured_carvers,
    //     "configured_carvers",
    //     "configured_carvers.txt"
    // );
    // export_category!(
    //     data.configured_features,
    //     "configured_features",
    //     "configured_features.txt"
    // );
    // export_category!(
    //     data.placed_features,
    //     "placed_features",
    //     "placed_features.txt"
    // );
    // export_category!(data.biomes, "biomes", "biomes.txt");
    // export_category!(
    //     data.structure_types,
    //     "structure_types",
    //     "structure_types.txt"
    // );
    // export_category!(data.structure_sets, "structure_sets", "structure_sets.txt");
    // export_category!(
    //     data.structure_template_pools,
    //     "structure_template_pools",
    //     "structure_template_pools.txt"
    // );
    // export_category!(
    //     data.structure_processor_lists,
    //     "structure_processor_lists",
    //     "structure_processor_lists.txt"
    // )
}
