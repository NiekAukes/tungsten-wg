use crate::parse::{
    MinecraftData, model::Density, model::DensityType, spline::SplineParseFunctions,
};

pub trait DensityParseFunctions<'m> {
    fn parse_density_function(&self, json_str: &str) -> Density<'m>;
    fn parse_density_function_from_value(&self, value: &serde_json::Value) -> Density<'m>;
    fn parse_density_function_from_value_and_name(
        &self,
        value: &serde_json::Value,
        canonical_name: &str,
    ) -> Density<'m>;
}

impl<'m> DensityParseFunctions<'m> for MinecraftData<'m> {
    fn parse_density_function(&self, json_str: &str) -> Density<'m> {
        // Implement parsing logic for DensityFunction
        // convert the JSON string to Values, and parse recursively
        let value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        self.parse_density_function_from_value(&value)
    }

    fn parse_density_function_from_value(&self, value: &serde_json::Value) -> Density<'m> {
        // Implement recursive parsing logic for DensityFunction from serde_json::Value
        // the value may be an object with a "type" field indicating the variant
        // or a primitive value for Const
        if let Some(obj) = value.as_object() {
            if let Some(type_value) = obj.get("type") {
                if let Some(type_str) = type_value.as_str() {
                    match type_str {
                        "minecraft:noise" => {
                            let noise = obj
                                .get("noise")
                                .and_then(|v| v.as_str())
                                .expect("Missing noise field in minecraft:noise");

                            // noise is already defined, so just reference it
                            let noise = self
                                .normal_noises
                                .get(noise)
                                .expect(&format!("Referenced noise not found: {}", noise));
                            let xz_scale =
                                obj.get("xz_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            let y_scale =
                                obj.get("y_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);

                            self.arena.alloc(DensityType::Noise(*noise))
                        }
                        "minecraft:cache_2d" => {
                            let source_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:cache_2d");
                            let source = self.parse_density_function_from_value(source_value);
                            // let xz_scale = obj.get("xz_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            // let y_scale = obj.get("y_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);

                            self.arena.alloc(DensityType::Cache2d { argument: source })
                        }

                        "minecraft:squeeze" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:squeeze");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::Squeeze { argument })
                        }
                        "minecraft:add" => {
                            let left_value = obj
                                .get("argument1")
                                .expect("Missing argument1 field in minecraft:add");
                            let right_value = obj
                                .get("argument2")
                                .expect("Missing argument2 field in minecraft:add");
                            let left = self.parse_density_function_from_value(left_value);
                            let right = self.parse_density_function_from_value(right_value);
                            self.arena.alloc(DensityType::Add { left, right })
                        }
                        "minecraft:mul" => {
                            let left_value = obj
                                .get("argument1")
                                .expect("Missing argument1 field in minecraft:add");
                            let right_value = obj
                                .get("argument2")
                                .expect("Missing argument2 field in minecraft:add");
                            let left = self.parse_density_function_from_value(left_value);
                            let right = self.parse_density_function_from_value(right_value);
                            self.arena.alloc(DensityType::Multiply { left, right })
                        }

                        "minecraft:interpolated" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:interpolated");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::Interpolated { argument })
                        }

                        "minecraft:blend_density" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:blend_density");
                            let argument = self.parse_density_function_from_value(argument_value);

                            // no blend_density is used, because it is for compatibility with older worlds
                            argument
                        }

                        "minecraft:end_islands" => self.arena.alloc(DensityType::EndIslands),
                        "minecraft:y_clamped_gradient" => {
                            let from_y = obj
                                .get("from_y")
                                .and_then(|v| v.as_f64())
                                .expect("Missing from_y field in minecraft:y_clamped_gradient");
                            let to_y = obj
                                .get("to_y")
                                .and_then(|v| v.as_f64())
                                .expect("Missing to_y field in minecraft:y_clamped_gradient");
                            let from_value = obj
                                .get("from_value")
                                .and_then(|v| v.as_f64())
                                .expect("Missing from_value field in minecraft:y_clamped_gradient");
                            let to_value = obj
                                .get("to_value")
                                .and_then(|v| v.as_f64())
                                .expect("Missing to_value field in minecraft:y_clamped_gradient");

                            self.arena.alloc(DensityType::YClampedGradient {
                                from_y,
                                to_y,
                                from_value,
                                to_value,
                            })
                        }

                        "minecraft:flat_cache" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:flat_cache");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::FlatCache { argument })
                        }

                        "minecraft:old_blended_noise" => {
                            let smear_scale_multiplier = obj
                                .get("smear_scale_multiplier")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0);
                            let xz_factor =
                                obj.get("xz_factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            let xz_scale =
                                obj.get("xz_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            let y_factor =
                                obj.get("y_factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            let y_scale =
                                obj.get("y_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);

                            self.arena.alloc(DensityType::OldBlendedNoise {
                                smear_scale_multiplier,
                                xz_factor,
                                xz_scale,
                                y_factor,
                                y_scale,
                            })
                        }

                        "minecraft:shifted_noise" => {
                            let noise = obj
                                .get("noise")
                                .and_then(|v| v.as_str())
                                .expect("Missing noise field in minecraft:shifted_noise");

                            // noise is already defined, so just reference it
                            let noise = self
                                .normal_noises
                                .get(noise)
                                .expect(&format!("Referenced noise not found: {}", noise));

                            let shift_x_value = obj
                                .get("shift_x")
                                .expect("Missing shift_x field in minecraft:shifted_noise");
                            let shift_x = self.parse_density_function_from_value(shift_x_value);

                            let shift_y =
                                obj.get("shift_y").and_then(|v| v.as_f64()).unwrap_or(0.0);

                            let shift_z_value = obj
                                .get("shift_z")
                                .expect("Missing shift_z field in minecraft:shifted_noise");
                            let shift_z = self.parse_density_function_from_value(shift_z_value);

                            let xz_scale =
                                obj.get("xz_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            let y_scale =
                                obj.get("y_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);

                            self.arena.alloc(DensityType::ShiftedNoise {
                                noise: *noise,
                                shift_x,
                                shift_y,
                                shift_z,
                                xz_scale,
                                y_scale,
                            })
                        }

                        "minecraft:shift_a" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:shift_a");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::ShiftA { argument })
                        }

                        "minecraft:shift_b" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:shift_b");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::ShiftB { argument })
                        }

                        "minecraft:blend_offset" | "minecraft:blend_alpha" => {
                            // blend_offset is used for compatibility with older worlds, it blends terrain heights near world borders
                            // for new terrian, it is effectively a 0
                            self.arena.alloc(DensityType::Const(0.0))
                        }

                        "minecraft:cache_once" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:cache_once");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::CacheOnce { argument })
                        }

                        "minecraft:spline" => {
                            let spline_value = obj
                                .get("spline")
                                .expect("Missing spline field in minecraft:spline");
                            let spline = self.parse_spline(spline_value);

                            self.arena.alloc(DensityType::Spline { spline })
                        }

                        "minecraft:abs" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:abs");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::Abs { argument })
                        }

                        "minecraft:min" => {
                            let left_value = obj
                                .get("argument1")
                                .expect("Missing argument1 field in minecraft:min");
                            let right_value = obj
                                .get("argument2")
                                .expect("Missing argument2 field in minecraft:min");
                            let left = self.parse_density_function_from_value(left_value);
                            let right = self.parse_density_function_from_value(right_value);
                            self.arena.alloc(DensityType::Min { left, right })
                        }
                        "minecraft:max" => {
                            let left_value = obj
                                .get("argument1")
                                .expect("Missing argument1 field in minecraft:max");
                            let right_value = obj
                                .get("argument2")
                                .expect("Missing argument2 field in minecraft:max");
                            let left = self.parse_density_function_from_value(left_value);
                            let right = self.parse_density_function_from_value(right_value);
                            self.arena.alloc(DensityType::Max { left, right })
                        }

                        "minecraft:range_choice" => {
                            let input_value = obj
                                .get("input")
                                .expect("Missing input field in minecraft:range_choice");
                            let input = self.parse_density_function_from_value(input_value);

                            let min_inclusive = obj
                                .get("min_inclusive")
                                .and_then(|v| v.as_f64())
                                .expect("Missing min_inclusive field in minecraft:range_choice");
                            let max_exclusive = obj
                                .get("max_exclusive")
                                .and_then(|v| v.as_f64())
                                .expect("Missing max_exclusive field in minecraft:range_choice");

                            let when_in_range_value = obj
                                .get("when_in_range")
                                .expect("Missing when_in_range field in minecraft:range_choice");
                            let when_in_range =
                                self.parse_density_function_from_value(when_in_range_value);

                            let when_out_of_range_value = obj.get("when_out_of_range").expect(
                                "Missing when_out_of_range field in minecraft:range_choice",
                            );
                            let when_out_of_range =
                                self.parse_density_function_from_value(when_out_of_range_value);

                            self.arena.alloc(DensityType::RangeChoice {
                                input,
                                min_inclusive,
                                max_exclusive,
                                when_in_range,
                                when_out_of_range,
                            })
                        }

                        "minecraft:quarter_negative" => {
                            // minecraft:quarter_negative is a modifier density function used in Minecraft's worldgen to scale and shift input values in a specific way.
                            // It's not commonly discussed outside the code, but its effect is mathematically simple.
                            // f(x) = min(x, 0) / 4

                            // this is easily represented as a min with 0, then a multiply by 0.25
                            // so no internal representation is needed
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:quarter_negative");
                            let argument = self.parse_density_function_from_value(argument_value);
                            let zero = self.arena.alloc(DensityType::Const(0.0));
                            let min = self.arena.alloc(DensityType::Min {
                                left: argument,
                                right: zero,
                            });
                            let factor = self.arena.alloc(DensityType::Const(0.25));
                            self.arena.alloc(DensityType::Multiply {
                                left: min,
                                right: factor,
                            })
                        }

                        "minecraft:half_negative" => {
                            // almost the same idea as quarter_negative, just with a weaker dampening effect.
                            // f(x) = x     if x >= 0
                            //        x / 2 if x <  0
                            // can be made via RangeChoice
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:half_negative");
                            let argument = self.parse_density_function_from_value(argument_value);
                            let half = self.arena.alloc(DensityType::Const(0.5));
                            let half_negative = self.arena.alloc(DensityType::Multiply {
                                left: argument,
                                right: half,
                            });
                            self.arena.alloc(DensityType::RangeChoice {
                                input: argument,
                                min_inclusive: -9999999999.0, // effectively negative infinity
                                max_exclusive: 0.0,
                                when_in_range: half_negative,
                                when_out_of_range: argument,
                            })
                        }

                        "minecraft:clamp" => {
                            let input_value = obj
                                .get("input")
                                .expect("Missing input field in minecraft:clamp");
                            let input = self.parse_density_function_from_value(input_value);

                            let min = obj
                                .get("min")
                                .and_then(|v| v.as_f64())
                                .expect("Missing min field in minecraft:clamp");
                            let max = obj
                                .get("max")
                                .and_then(|v| v.as_f64())
                                .expect("Missing max field in minecraft:clamp");

                            self.arena.alloc(DensityType::Clamp { input, min, max })
                        }

                        "minecraft:weird_scaled_sampler" => {
                            let input_value = obj
                                .get("input")
                                .expect("Missing input field in minecraft:weird_scaled_sampler");
                            let input = self.parse_density_function_from_value(input_value);

                            let noise_to_sample_str = obj
                                .get("noise")
                                .and_then(|v| v.as_str())
                                .expect("Missing noise field in minecraft:weird_scaled_sampler");
                            let noise_to_sample =
                                self.normal_noises.get(noise_to_sample_str).expect(&format!(
                                    "Referenced noise not found: {}",
                                    noise_to_sample_str
                                ));

                            let rarity_value_mapper = obj
                                .get("rarity_value_mapper")
                                .and_then(|v| v.as_str())
                                .expect("Missing rarity_value_mapper field in minecraft:weird_scaled_sampler")
                                .to_string();

                            self.arena.alloc(DensityType::WeirdScaledSampler {
                                input,
                                noise_to_sample: *noise_to_sample,
                                rarity_value_mapper,
                            })
                        }

                        "minecraft:square" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:square");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::Square { argument })
                        }

                        "minecraft:cube" => {
                            let argument_value = obj
                                .get("argument")
                                .expect("Missing argument field in minecraft:cube");
                            let argument = self.parse_density_function_from_value(argument_value);

                            self.arena.alloc(DensityType::Cube { argument })
                        }

                        _ => panic!(
                            "Invalid type field in DensityFunction: {}, with object: {:?}",
                            type_str, obj
                        ),
                    }
                } else {
                    panic!("Invalid type field in DensityFunction: {:?}", type_value);
                }
            } else {
                panic!("Missing type field in DensityFunction object: {:?}", obj);
            }
        } else if let Some(num) = value.as_f64() {
            //density_arena.intern(DensityFunctionType::Const(num))
            self.arena.alloc(DensityType::Const(num))
        } else {
            let s = value
                .as_str()
                .expect("DensityFunction must be an object, number, or string");
            // may be a reference to another density function
            if let Some(referenced) = self.density_functions.get(s) {
                *referenced
            } else if let Some(referenced) = self.normal_noises.get(s) {
                // also allow referencing normal noises directly
                self.arena.alloc(DensityType::Noise(*referenced))
            } else if let Some(referenced) = self.raw_data.density_functions.get(s) {
                // also allow referencing density functions directly
                let parsed = self.parse_density_function(referenced);
                let name = s.to_string();
                let argument = if let DensityType::FlatCache { argument } = parsed {
                    // turn the flat cache into a named density reference
                    *argument
                } else {
                    parsed
                };
                self.arena.alloc(DensityType::NamedDensityReference {
                    name: self.arena.alloc(name),
                    argument,
                })
            } else {
                panic!("Unknown DensityFunction type or reference: {:?}", value);
            }
        }
    }

    fn parse_density_function_from_value_and_name(
        &self,
        value: &serde_json::Value,
        canonical_name: &str,
    ) -> Density<'m> {
        let mut density = self.parse_density_function_from_value(value);
        if let DensityType::NamedDensityReference { argument, .. } = density {
            // if the density is already a named reference, just return it
            density = *argument;
        }
        // otherwise, create a new named reference with the given canonical name
        let name = self.arena.alloc(canonical_name.to_string());
        self.arena.alloc(DensityType::NamedDensityReference {
            name,
            argument: density,
        })
    }
}
