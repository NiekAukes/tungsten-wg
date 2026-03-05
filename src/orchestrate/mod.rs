pub mod dot;
pub mod model;
pub mod transform;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scale {
    x: i32,
    y: i32,
    z: i32,
}

impl Scale {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        // convert to fixed point with 16 bits of fractional precision
        Self {
            x: (x * 256.0) as i32,
            y: (y * 256.0) as i32,
            z: (z * 256.0) as i32,
        }
    }

    pub fn as_float(&self) -> (f32, f32, f32) {
        (
            self.x as f32 / 256.0,
            self.y as f32 / 256.0,
            self.z as f32 / 256.0,
        )
    }
}
