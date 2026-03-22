use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::parse::Interned;

#[derive(Debug, Clone)]
pub struct NoiseRouter<'m> {
    pub barrier: DensitySource<'m>,
    pub continents: DensitySource<'m>,
    pub depth: DensitySource<'m>,
    pub erosion: DensitySource<'m>,
    pub final_density: DensitySource<'m>,
    pub fluid_level_floodedness: DensitySource<'m>,
    pub fluid_level_spread: DensitySource<'m>,
    pub initial_density_without_jaggedness: DensitySource<'m>,
    pub lava: DensitySource<'m>,
    pub ridges: DensitySource<'m>,
    pub temperature: DensitySource<'m>,
    pub vegetation: DensitySource<'m>,
    pub vein_gap: DensitySource<'m>,
    pub vein_ridged: DensitySource<'m>,
    pub vein_toggle: DensitySource<'m>,
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

#[derive(Debug, Clone, Copy)]
pub enum DensitySource<'m> {
    MultiSamplingDensity {
        density: Density<'m>,
        dimensions: (i32, i32, i32),
    },
    SingleSamplingDensity {
        density: Density<'m>,
    },
}

pub type Density<'m> = Interned<'m, DensityType<'m>>;

#[derive(Debug, Clone, PartialEq)]
pub enum DensityType<'m> {
    Const(f64),
    Noise {
        name: String,
        noise: NormalNoise<'m>,
        xz_scale: f64,
        y_scale: f64,
    },
    Add {
        left: Density<'m>,
        right: Density<'m>,
    },
    Multiply {
        left: Density<'m>,
        right: Density<'m>,
    },
    Cache2d {
        argument: Density<'m>,
    },
    Squeeze {
        argument: Density<'m>,
    },
    Interpolated {
        argument: Density<'m>,
    },
    EndIslands,
    YClampedGradient {
        // Minecraft 1.20+
        from_y: f64,
        to_y: f64,
        from_value: f64,
        to_value: f64,
    },

    FlatCache {
        argument: Density<'m>,
    },

    OldBlendedNoise {
        smear_scale_multiplier: f64,
        xz_factor: f64,
        xz_scale: f64,
        y_factor: f64,
        y_scale: f64,
    },

    ShiftedNoise {
        name: String,
        noise: NormalNoise<'m>,
        shift_x: Density<'m>,
        shift_y: Density<'m>,
        shift_z: Density<'m>,
        xz_scale: f64,
        y_scale: f64,
    },

    // minecraft:shift_a is one of the coordinate warping density functions used in terrain generation.
    // It doesn't directly produce terrain heights, instead it shifts the input coordinates before another noise function samples them.
    // In effect: instead of sampling noise(x, y, z), the system samples noise(x + offset, y, z + offset)
    // where the offset comes from a secondary "shift" noise.
    ShiftA {
        argument: NormalNoise<'m>,
        name: String,
    },

    // Similar to ShiftA, but shift_b applies different offsets to the coordinates.
    ShiftB {
        argument: NormalNoise<'m>,
        name: String,
    },

    // marks the function as pure, and storing the result to be reused inside a computation
    CacheOnce {
        argument: Density<'m>,
    },

    Spline {
        spline: Spline<'m>,
    },

    Abs {
        argument: Density<'m>,
    },

    Min {
        left: Density<'m>,
        right: Density<'m>,
    },
    Max {
        left: Density<'m>,
        right: Density<'m>,
    },

    // if (min <= input <= max) return value_a;
    // else return value_b;
    RangeChoice {
        input: Density<'m>,
        min_inclusive: f64,
        max_exclusive: f64,
        when_in_range: Density<'m>,
        when_out_of_range: Density<'m>,
    },

    Clamp {
        input: Density<'m>,
        min: f64,
        max: f64,
    },

    WeirdScaledSampler {
        input: Density<'m>,
        noise_to_sample: NormalNoise<'m>,
        rarity_value_mapper: String,
    },

    Square {
        argument: Density<'m>,
    },

    Cube {
        argument: Density<'m>,
    },
    NamedDensityReference {
        name: Interned<'m, String>,
        argument: Density<'m>,
    },
}

impl Hash for DensityType<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            DensityType::Const(val) => {
                0.hash(state);
                val.to_bits().hash(state);
            }
            DensityType::Noise {
                name,
                noise,
                xz_scale,
                y_scale,
            } => {
                1.hash(state);
                name.hash(state);
                noise.hash(state);
                xz_scale.to_bits().hash(state);
                y_scale.to_bits().hash(state);
            }
            DensityType::Add { left, right } => {
                2.hash(state);
                left.hash(state);
                right.hash(state);
            }
            DensityType::Multiply { left, right } => {
                3.hash(state);
                left.hash(state);
                right.hash(state);
            }
            DensityType::Cache2d { argument: source } => {
                4.hash(state);
                source.hash(state);
            }
            DensityType::Squeeze { argument } => {
                5.hash(state);
                argument.hash(state);
            }
            DensityType::Interpolated { argument } => {
                6.hash(state);
                argument.hash(state);
            }
            DensityType::EndIslands => {
                8.hash(state);
            }
            DensityType::YClampedGradient {
                from_y,
                to_y,
                from_value,
                to_value,
            } => {
                9.hash(state);
                from_y.to_bits().hash(state);
                to_y.to_bits().hash(state);
                from_value.to_bits().hash(state);
                to_value.to_bits().hash(state);
            }

            DensityType::FlatCache { argument } => {
                10.hash(state);
                argument.hash(state);
            }

            DensityType::OldBlendedNoise {
                smear_scale_multiplier,
                xz_factor,
                xz_scale,
                y_factor,
                y_scale,
            } => {
                11.hash(state);
                smear_scale_multiplier.to_bits().hash(state);
                xz_factor.to_bits().hash(state);
                xz_scale.to_bits().hash(state);
                y_factor.to_bits().hash(state);
                y_scale.to_bits().hash(state);
            }

            DensityType::ShiftedNoise {
                name,
                noise,
                shift_x,
                shift_y,
                shift_z,
                xz_scale,
                y_scale,
            } => {
                12.hash(state);
                name.hash(state);
                noise.hash(state);
                shift_x.hash(state);
                shift_y.hash(state);
                shift_z.hash(state);
                xz_scale.to_bits().hash(state);
                y_scale.to_bits().hash(state);
            }

            DensityType::ShiftA { argument, name } => {
                13.hash(state);
                argument.hash(state);
                name.hash(state);
            }

            DensityType::ShiftB { argument, name } => {
                14.hash(state);
                argument.hash(state);
                name.hash(state);
            }
            DensityType::CacheOnce { argument } => {
                15.hash(state);
                argument.hash(state);
            }
            DensityType::Spline { spline } => {
                16.hash(state);
                spline.hash(state);
            }
            DensityType::Abs { argument } => {
                17.hash(state);
                argument.hash(state);
            }
            DensityType::Min { left, right } => {
                18.hash(state);
                left.hash(state);
                right.hash(state);
            }
            DensityType::Max { left, right } => {
                19.hash(state);
                left.hash(state);
                right.hash(state);
            }
            DensityType::RangeChoice {
                input,
                min_inclusive,
                max_exclusive,
                when_in_range,
                when_out_of_range,
            } => {
                20.hash(state);
                input.hash(state);
                min_inclusive.to_bits().hash(state);
                max_exclusive.to_bits().hash(state);
                when_in_range.hash(state);
                when_out_of_range.hash(state);
            }

            DensityType::Clamp { input, min, max } => {
                21.hash(state);
                input.hash(state);
                min.to_bits().hash(state);
                max.to_bits().hash(state);
            }

            DensityType::WeirdScaledSampler {
                input,
                noise_to_sample,
                rarity_value_mapper,
            } => {
                22.hash(state);
                input.hash(state);
                noise_to_sample.hash(state);
                rarity_value_mapper.hash(state);
            }

            DensityType::Square { argument } => {
                23.hash(state);
                argument.hash(state);
            }

            DensityType::Cube { argument } => {
                24.hash(state);
                argument.hash(state);
            }
            DensityType::NamedDensityReference { name, argument } => {
                25.hash(state);
                name.hash(state);
                argument.hash(state);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NormalNoiseType {
    // Define fields as per Minecraft's JSON structure
    #[serde(rename = "firstOctave")]
    pub first_octave: i32,
    pub amplitudes: Vec<f64>,
}

impl Eq for NormalNoiseType {}

impl Hash for NormalNoiseType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.first_octave.hash(state);
        for amp in &self.amplitudes {
            amp.to_bits().hash(state);
        }
    }
}

pub type NormalNoise<'m> = Interned<'m, NormalNoiseType>;

pub type Spline<'m> = Interned<'m, SplineType<'m>>;

#[derive(PartialEq, Hash, Debug)]
pub struct SplineType<'m> {
    pub(crate) coordinate: Density<'m>,
    pub(crate) spline_points: &'m [SplinePoint<'m>],
}

#[derive(PartialEq, Debug, Clone)]
pub struct SplinePoint<'m> {
    pub(crate) derivative: f64,
    pub(crate) location: f64,
    pub(crate) value: SplineValue<'m>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum SplineValue<'m> {
    Spline(Spline<'m>),
    Const(f64),
}
impl<'m> Eq for DensityType<'m> {}

impl<'m> Eq for SplineValue<'m> {}

impl<'m> Hash for SplinePoint<'m> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.derivative.to_bits().hash(state);
        self.location.to_bits().hash(state);
        self.value.hash(state);
    }
}

impl<'m> Hash for SplineValue<'m> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            SplineValue::Const(c) => {
                c.to_bits().hash(state);
            }
            SplineValue::Spline(spline) => {
                spline.hash(state);
            }
        }
    }
}

impl<'m> DensitySource<'m> {
    pub fn get_density(&self) -> &Density<'m> {
        match self {
            DensitySource::MultiSamplingDensity { density, .. } => density,
            DensitySource::SingleSamplingDensity { density } => density,
        }
    }
}
