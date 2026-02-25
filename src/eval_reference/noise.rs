use crate::{eval_reference::Evaluate, parse::model::NormalNoise};

impl<'m> Evaluate for NormalNoise<'m> {
    fn evaluate(&self, x: f64, y: f64, z: f64, _memory: &mut [f64]) -> f64 {
        // For simplicity, we'll use a basic noise function here.
        // this is not a real noise function, just a placeholder
        x.sin() * y.cos() * z.sin()
    }
}

pub fn old_blended_noise(
    x: f64,
    y: f64,
    z: f64,
    smear_scale_multiplier: f64,
    xz_factor: f64,
    xz_scale: f64,
    y_factor: f64,
    y_scale: f64,
) -> f64 {
    // Placeholder implementation for old blended noise
    let smear = smear_scale_multiplier * (x * xz_factor).sin() * (z * xz_factor).cos();
    let height = (y * y_factor).sin() * y_scale;
    smear + height
}
