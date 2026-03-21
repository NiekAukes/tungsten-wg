pub mod dot;
pub mod model;
pub mod transform;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scale {
    x: u32,
    y: u32,
    z: u32,
}

impl Scale {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        // convert to fixed point with 16 bits of fractional precision
        Self {
            x: (x * 65565.0) as u32,
            y: (y * 65565.0) as u32,
            z: (z * 65565.0) as u32,
        }
    }

    pub fn as_float(&self) -> (f32, f32, f32) {
        (
            self.x as f32 / 65565.0,
            self.y as f32 / 65565.0,
            self.z as f32 / 65565.0,
        )
    }
}

pub trait Flatten {
    type Output;

    fn flatten(&self) -> Self::Output;
}

impl Flatten for (i32, i32, i32) {
    type Output = i32;

    fn flatten(&self) -> Self::Output {
        self.0 * self.1 * self.2
    }
}

impl From<(i32, i32, i32)> for Scale {
    fn from(value: (i32, i32, i32)) -> Self {
        Self {
            x: value.0 as u32,
            y: value.1 as u32,
            z: value.2 as u32,
        }
    }
}

impl From<(f32, f32, f32)> for Scale {
    fn from(value: (f32, f32, f32)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl Default for Scale {
    fn default() -> Self {
        Self {
            x: 65565,
            y: 65565,
            z: 65565,
        }
    }
}
