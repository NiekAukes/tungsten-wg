use std::hash::Hash;
use std::hash::Hasher;

pub mod dot;
pub mod model;
pub mod transform;

#[derive(Debug, Clone, Copy)]
pub struct Scale {
    x: u32,
    y: u32,
    z: u32,
    rx: f64,
    ry: f64,
    rz: f64,
}

impl Scale {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        // convert to fixed point with 16 bits of fractional precision
        Self {
            x: (x * 65565.0) as u32,
            y: (y * 65565.0) as u32,
            z: (z * 65565.0) as u32,
            rx: x,
            ry: y,
            rz: z,
        }
    }

    pub fn as_float(&self) -> (f32, f32, f32) {
        (self.rx as f32, self.ry as f32, self.rz as f32)
    }

    pub fn as_int(&self) -> (u32, u32, u32) {
        (self.x, self.y, self.z)
    }
}

pub trait Flatten {
    type Output;

    fn flatten(&self) -> Self::Output;
}

impl Flatten for (i32, i32, i32) {
    type Output = usize;

    fn flatten(&self) -> Self::Output {
        self.0 as usize * self.1 as usize * self.2 as usize
    }
}

impl From<(i32, i32, i32)> for Scale {
    fn from(value: (i32, i32, i32)) -> Self {
        Self {
            x: value.0 as u32,
            y: value.1 as u32,
            z: value.2 as u32,
            rx: value.0 as f64 / 65565.0,
            ry: value.1 as f64 / 65565.0,
            rz: value.2 as f64 / 65565.0,
        }
    }
}

impl From<(f32, f32, f32)> for Scale {
    fn from(value: (f32, f32, f32)) -> Self {
        Self::new(value.0 as f64, value.1 as f64, value.2 as f64)
    }
}

impl From<(f64, f64, f64)> for Scale {
    fn from(value: (f64, f64, f64)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl Default for Scale {
    fn default() -> Self {
        Self {
            x: 65565,
            y: 65565,
            z: 65565,
            rx: 1.0,
            ry: 1.0,
            rz: 1.0,
        }
    }
}

impl Hash for Scale {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.z.hash(state);
    }
}

impl PartialEq for Scale {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl Eq for Scale {}
