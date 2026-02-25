use core::panic;
use std::{
    collections::HashMap,
    fmt::{Debug, Formatter},
    hash::Hash,
};

use bumpalo::Bump;
use serde::{Deserialize, Serialize, ser::Error};

use crate::{
    config_load::MinecraftDataRaw,
    parse::{
        density::DensityParseFunctions,
        model::Spline,
        model::{Density, NormalNoise, NormalNoiseType},
    },
};

pub mod density;
pub mod model;
pub mod pretty;
pub mod spline;
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

/*
{
  "aquifers_enabled": true,
  "default_block": {
    "Name": "minecraft:stone"
  },
  "default_fluid": {
    "Name": "minecraft:water",
    "Properties": {
      "level": "0"
    }
  },
  "disable_mob_generation": false,
  "legacy_random_source": false,
  "noise": {
    "height": 384,
    "min_y": -64,
    "size_horizontal": 1,
    "size_vertical": 2
  },
  "noise_router": {
    "barrier": {
      "type": "minecraft:noise",
      "noise": "minecraft:aquifer_barrier",
      "xz_scale": 1.0,
      "y_scale": 0.5
    },
    "continents": "minecraft:overworld/continents",
    "depth": "minecraft:overworld/depth",
    "erosion": "minecraft:overworld/erosion",
    "final_density": { ... },
    "fluid_level_floodedness": {
      "type": "minecraft:noise",
      "noise": "minecraft:aquifer_fluid_level_floodedness",
      "xz_scale": 1.0,
      "y_scale": 0.67
    },
    "fluid_level_spread": {
      "type": "minecraft:noise",
      "noise": "minecraft:aquifer_fluid_level_spread",
      "xz_scale": 1.0,
      "y_scale": 0.7142857142857143
    },
    "initial_density_without_jaggedness": { ... },
    "lava": {
      "type": "minecraft:noise",
      "noise": "minecraft:aquifer_lava",
      "xz_scale": 1.0,
      "y_scale": 1.0
    },
    "ridges": "minecraft:overworld/ridges",
    "temperature": {
      "type": "minecraft:shifted_noise",
      "noise": "minecraft:temperature",
      "shift_x": "minecraft:shift_x",
      "shift_y": 0.0,
      "shift_z": "minecraft:shift_z",
      "xz_scale": 0.25,
      "y_scale": 0.0
    },
    "vegetation": {
      "type": "minecraft:shifted_noise",
      "noise": "minecraft:vegetation",
      "shift_x": "minecraft:shift_x",
      "shift_y": 0.0,
      "shift_z": "minecraft:shift_z",
      "xz_scale": 0.25,
      "y_scale": 0.0
    },
    "vein_gap": {
      "type": "minecraft:noise",
      "noise": "minecraft:ore_gap",
      "xz_scale": 1.0,
      "y_scale": 1.0
    },
    "vein_ridged": { ... },
    "vein_toggle": {
      "type": "minecraft:interpolated",
      "argument": {
        "type": "minecraft:range_choice",
        "input": "minecraft:y",
        "max_exclusive": 51.0,
        "min_inclusive": -60.0,
        "when_in_range": {
          "type": "minecraft:noise",
          "noise": "minecraft:ore_veininess",
          "xz_scale": 1.5,
          "y_scale": 1.5
        },
        "when_out_of_range": 0.0
      }
    }
  },
*/

#[derive(Debug, Clone)]
pub struct NoiseRouter<'m> {
    pub barrier: Density<'m>,
    pub continents: Density<'m>,
    pub depth: Density<'m>,
    pub erosion: Density<'m>,
    pub final_density: Density<'m>,
    pub fluid_level_floodedness: Density<'m>,
    pub fluid_level_spread: Density<'m>,
    pub initial_density_without_jaggedness: Density<'m>,
    pub lava: Density<'m>,
    pub ridges: Density<'m>,
    pub temperature: Density<'m>,
    pub vegetation: Density<'m>,
    pub vein_gap: Density<'m>,
    pub vein_ridged: Density<'m>,
    pub vein_toggle: Density<'m>,
}

#[derive(Debug, Clone)]
pub struct NoiseGeneratorSettings<'m> {
    pub aquifers_enabled: bool,
    pub default_block: String,
    pub default_fluid: String,
    pub default_fluid_level: i32,
    pub disable_mob_generation: bool,
    pub noise: NoiseSettings,
    pub noise_router: NoiseRouter<'m>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NoiseSettings {
    pub height: i32,
    pub min_y: i32,
    pub size_horizontal: i32,
    pub size_vertical: i32,
}

// #[derive(Serialize, Deserialize, Debug, Clone)]
// pub struct SlideSettings {
//     pub target: i32,
//     pub size: i32,
//     pub offset: i32,
// }

pub type Interned<'m, T> = &'m T; // TODO intern later

pub struct MinecraftData<'m> {
    arena: &'m Bump,
    raw_data: &'m MinecraftDataRaw,
    pub noise_settings: HashMap<String, NoiseGeneratorSettings<'m>>,
    pub density_functions: HashMap<String, Density<'m>>,
    pub normal_noises: HashMap<String, NormalNoise<'m>>,
}

impl<'m> Debug for MinecraftData<'m> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinecraftData")
            .field("noise_settings", &self.noise_settings)
            .field("density_functions", &self.density_functions)
            .field("normal_noises", &self.normal_noises)
            .finish()
    }
}

impl<'m> MinecraftData<'m> {
    pub fn new(arena: &'m Bump, raw: &'m MinecraftDataRaw) -> MinecraftData<'m> {
        MinecraftData {
            arena: arena,
            raw_data: raw,
            noise_settings: HashMap::new(),
            density_functions: HashMap::new(),
            normal_noises: HashMap::new(),
        }
    }

    pub fn parse_from_raw(&mut self) {
        let raw = &self.raw_data;
        for (name, json_str) in &raw.normal_noises {
            let parsed: NormalNoiseType = serde_json::from_str(&json_str).unwrap();
            let noise = self.arena.alloc(parsed);
            println!("Parsed NormalNoise: {:?}", name);
            self.normal_noises.insert(name.to_string(), noise);
        }

        for (name, json_str) in &raw.noise_settings {
            // let parsed = { self.parse_noise_settings(&json_str).unwrap() };
            // self.noise_settings.insert(name, parsed);
            let parsed = {
                let tmp = self.parse_noise_settings(&json_str).unwrap();
                tmp
            };
            self.noise_settings.insert(name.to_string(), parsed);
        }

        // parsing these is not necessarily needed, as they are just imports
        // but it's useful for debugging
        for (name, json_str) in &raw.density_functions {
            let parsed = self.parse_density_function(&json_str);
            self.density_functions.insert(name.to_string(), parsed);
        }
    }

    fn parse_noise_settings(
        &self,
        json_str: &str,
    ) -> Result<NoiseGeneratorSettings<'m>, serde_json::Error> {
        let value: serde_json::Value = serde_json::from_str(json_str)?;
        let aquifers_enabled = value
            .get("aquifers_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let default_block = value
            .get("default_block")
            .and_then(|v| v.get("Name"))
            .and_then(|v| v.as_str())
            .expect("default_block.Name missing");
        let (default_fluid, default_fluid_level) = if let Some(fluid) = value.get("default_fluid") {
            let name = fluid
                .get("Name")
                .and_then(|v| v.as_str())
                .expect("default_fluid.Name missing");
            let level = fluid
                .get("Properties")
                .and_then(|v| v.get("level"))
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            (name.to_string(), level)
        } else {
            panic!("default_fluid missing");
        };
        let disable_mob_generation = value
            .get("disable_mob_generation")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // legacy_random_source is ignored

        let noise_value = value.get("noise").expect("noise field missing");
        let noise: NoiseSettings = serde_json::from_value(noise_value.clone())?;
        let noise_router_value = value
            .get("noise_router")
            .expect("noise_router field missing");
        let noise_router = self.parse_noise_router(noise_router_value);

        Ok(NoiseGeneratorSettings {
            aquifers_enabled,
            default_block: default_block.to_string(),
            default_fluid,
            default_fluid_level,
            disable_mob_generation,
            noise,
            noise_router: noise_router,
        })
    }

    fn parse_noise_router(&self, value: &serde_json::Value) -> NoiseRouter<'m> {
        let barrier = if let Some(barrier_value) = value.get("barrier") {
            self.parse_density_function_from_value_and_name(barrier_value, "barrier")
        } else {
            panic!("Missing barrier field in noise_router")
        };
        let continents = if let Some(continents_value) = value.get("continents") {
            self.parse_density_function_from_value_and_name(continents_value, "continents")
        } else {
            panic!("Missing continents field in noise_router")
        };
        let depth = if let Some(depth_value) = value.get("depth") {
            self.parse_density_function_from_value_and_name(depth_value, "depth")
        } else {
            panic!("Missing depth field in noise_router")
        };
        let erosion = if let Some(erosion_value) = value.get("erosion") {
            self.parse_density_function_from_value_and_name(erosion_value, "erosion")
        } else {
            panic!("Missing erosion field in noise_router")
        };
        let final_density = if let Some(final_density_value) = value.get("final_density") {
            self.parse_density_function_from_value_and_name(final_density_value, "final_density")
        } else {
            panic!("Missing final_density field in noise_router")
        };
        let fluid_level_floodedness =
            if let Some(fluid_level_floodedness_value) = value.get("fluid_level_floodedness") {
                self.parse_density_function_from_value_and_name(
                    fluid_level_floodedness_value,
                    "fluid_level_floodedness",
                )
            } else {
                panic!("Missing fluid_level_floodedness field in noise_router")
            };
        let fluid_level_spread =
            if let Some(fluid_level_spread_value) = value.get("fluid_level_spread") {
                self.parse_density_function_from_value_and_name(
                    fluid_level_spread_value,
                    "fluid_level_spread",
                )
            } else {
                panic!("Missing fluid_level_spread field in noise_router")
            };
        let initial_density_without_jaggedness =
            if let Some(initial_density_without_jaggedness_value) =
                value.get("initial_density_without_jaggedness")
            {
                self.parse_density_function_from_value_and_name(
                    initial_density_without_jaggedness_value,
                    "initial_density_without_jaggedness",
                )
            } else {
                panic!("Missing initial_density_without_jaggedness field in noise_router")
            };
        let lava = if let Some(lava_value) = value.get("lava") {
            self.parse_density_function_from_value_and_name(lava_value, "lava")
        } else {
            panic!("Missing lava field in noise_router")
        };
        let ridges = if let Some(ridges_value) = value.get("ridges") {
            self.parse_density_function_from_value_and_name(ridges_value, "ridges")
        } else {
            panic!("Missing ridges field in noise_router")
        };
        let temperature = if let Some(temperature_value) = value.get("temperature") {
            self.parse_density_function_from_value_and_name(temperature_value, "temperature")
        } else {
            panic!("Missing temperature field in noise_router")
        };
        let vegetation = if let Some(vegetation_value) = value.get("vegetation") {
            self.parse_density_function_from_value_and_name(vegetation_value, "vegetation")
        } else {
            panic!("Missing vegetation field in noise_router")
        };
        let vein_gap = if let Some(vein_gap_value) = value.get("vein_gap") {
            self.parse_density_function_from_value_and_name(vein_gap_value, "vein_gap")
        } else {
            panic!("Missing vein_gap field in noise_router")
        };
        let vein_ridged = if let Some(vein_ridged_value) = value.get("vein_ridged") {
            self.parse_density_function_from_value_and_name(vein_ridged_value, "vein_ridged")
        } else {
            panic!("Missing vein_ridged field in noise_router")
        };
        let vein_toggle = if let Some(vein_toggle_value) = value.get("vein_toggle") {
            self.parse_density_function_from_value_and_name(vein_toggle_value, "vein_toggle")
        } else {
            panic!("Missing vein_toggle field in noise_router")
        };
        NoiseRouter {
            barrier,
            continents,
            depth,
            erosion,
            final_density,
            fluid_level_floodedness,
            fluid_level_spread,
            initial_density_without_jaggedness,
            lava,
            ridges,
            temperature,
            vegetation,
            vein_gap,
            vein_ridged,
            vein_toggle,
        }
    }
}

impl<'m> NoiseRouter<'m> {
    pub fn all_densities(&self) -> Vec<Density<'m>> {
        vec![
            self.barrier,
            self.continents,
            self.depth,
            self.erosion,
            self.final_density,
            self.fluid_level_floodedness,
            self.fluid_level_spread,
            self.initial_density_without_jaggedness,
            self.lava,
            self.ridges,
            self.temperature,
            self.vegetation,
            self.vein_gap,
            self.vein_ridged,
            self.vein_toggle,
        ]
    }
}
