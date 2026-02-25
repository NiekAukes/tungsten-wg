use crate::parse::model::{Density, DensityType};

pub mod noise;

pub trait Evaluate {
    fn evaluate(&self, x: f64, y: f64, z: f64, memory: &mut [f64]) -> f64;
}

pub struct Evaluator<'m> {
    pub density_function: Density<'m>,
    pub memory: Vec<f64>,
}

impl<'m> Evaluator<'m> {
    pub fn new(density_function: Density<'m>) -> Self {
        let memory_size = 1000; //density_function.max_memory_index() + 1;
        Evaluator {
            density_function,
            memory: vec![0.0; memory_size],
        }
    }
}

impl Evaluate for Evaluator<'_> {
    fn evaluate(&self, x: f64, y: f64, z: f64, memory: &mut [f64]) -> f64 {
        self.density_function.evaluate(x, y, z, memory)
    }
}

impl Evaluate for Density<'_> {
    fn evaluate(&self, x: f64, y: f64, z: f64, memory: &mut [f64]) -> f64 {
        match *self {
            DensityType::Const(c) => *c,
            DensityType::Noise(normal_noise_type) => normal_noise_type.evaluate(x, y, z, memory),
            DensityType::Add { left, right } => {
                left.evaluate(x, y, z, memory) + right.evaluate(x, y, z, memory)
            }
            DensityType::Multiply { left, right } => {
                left.evaluate(x, y, z, memory) * right.evaluate(x, y, z, memory)
            }
            DensityType::Cache2d { argument } => {
                argument.evaluate(x, 0.0, z, memory)
                // TODO: implement 2D caching
            }
            DensityType::Squeeze { argument } => {
                let a = argument.evaluate(x, y, z, memory);
                // sign(a) * sqrt(abs(a))
                if a < 0.0 { -(-a).sqrt() } else { a.sqrt() }
            }
            DensityType::Interpolated { argument } => {
                // TODO: implement interpolation
                argument.evaluate(x, y, z, memory)
            }
            DensityType::EndIslands => todo!(),
            DensityType::YClampedGradient {
                from_y,
                to_y,
                from_value,
                to_value,
            } => {
                if y <= *from_y {
                    *from_value
                } else if y >= *to_y {
                    *to_value
                } else {
                    let t = (y - from_y) / (to_y - from_y);
                    from_value + t * (to_value - from_value)
                }
            }
            DensityType::FlatCache { argument } => {
                // TODO: implement flat caching
                argument.evaluate(x, y, z, memory)
            }
            DensityType::OldBlendedNoise {
                smear_scale_multiplier,
                xz_factor,
                xz_scale,
                y_factor,
                y_scale,
            } => noise::old_blended_noise(
                x,
                y,
                z,
                *smear_scale_multiplier,
                *xz_factor,
                *xz_scale,
                *y_factor,
                *y_scale,
            ),
            DensityType::ShiftedNoise {
                noise,
                shift_x,
                shift_y,
                shift_z,
                xz_scale,
                y_scale,
            } => {
                let shift_x_val = shift_x.evaluate(x, y, z, memory);

                let shift_z_val = shift_z.evaluate(x, y, z, memory);
                // not sure if this is correct
                let new_x = x + shift_x_val * xz_scale;
                let new_y = y + shift_y * y_scale;
                let new_z = z + shift_z_val * xz_scale;
                noise.evaluate(new_x, new_y, new_z, memory)
            }
            DensityType::ShiftA { argument } => {
                let shift = argument.evaluate(x, y, z, memory);
                shift
            }
            DensityType::ShiftB { argument } => todo!(),
            DensityType::CacheOnce { argument } => todo!(),
            DensityType::Spline { spline } => todo!(),
            DensityType::Abs { argument } => todo!(),
            DensityType::Min { left, right } => todo!(),
            DensityType::Max { left, right } => todo!(),
            DensityType::RangeChoice {
                input,
                min_inclusive,
                max_exclusive,
                when_in_range,
                when_out_of_range,
            } => todo!(),
            DensityType::Clamp { input, min, max } => todo!(),
            DensityType::WeirdScaledSampler {
                input,
                noise_to_sample,
                rarity_value_mapper,
            } => todo!(),
            DensityType::Square { argument } => todo!(),
            DensityType::Cube { argument } => todo!(),
            DensityType::NamedDensityReference { name, argument } => todo!(),
        }
    }
}
