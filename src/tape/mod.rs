pub mod lower;

// A DensityTape is a linear representation of a density function
#[derive(Debug)]
pub struct DensityBody {
    cell_size: (u8, u8, u8),
    instructions: Vec<DensityInstruction>,
}

type DensityInstructionIndex = u32;

#[derive(Debug)]
pub enum DensityInstruction {
    Constant(f64),
    Noise4 {
        first_octave: i32,
        amplitudes: [f64; 4],
        length: u8,
    },

    NoiseGeneral {
        first_octave: i32,
        amplitudes: Vec<f64>,
    },

    Add(DensityInstructionIndex, DensityInstructionIndex),
    Multiply(DensityInstructionIndex, DensityInstructionIndex),
    Min(DensityInstructionIndex, DensityInstructionIndex),
    Max(DensityInstructionIndex, DensityInstructionIndex),
    Abs(DensityInstructionIndex),
    Square(DensityInstructionIndex),
    Cube(DensityInstructionIndex),

    Interpolate {
        tape: DensityBody,
    },

    Cache2D {
        tape: DensityBody,
    },
    EndIslands,
    YClampedGradient {
        // Minecraft 1.20+
        from_y: f64,
        to_y: f64,
        from_value: f64,
        to_value: f64,
    },

    OldBlendedNoise {
        smear_scale_multiplier: f64,
        xz_factor: f64,
        xz_scale: f64,
        y_factor: f64,
        y_scale: f64,
    },

    ShiftedNoise {
        noise: DensityInstructionIndex,
        shift_x: DensityInstructionIndex,
        shift_z: DensityInstructionIndex,
        shift_y: f64,
        xz_scale: f64,
        y_scale: f64,
    },

    ShiftA {
        argument: DensityInstructionIndex,
    },

    ShiftB {
        argument: DensityInstructionIndex,
    },

    // cacheonce is used to memoize the result of a density function for the duration of a single generation step.
    CacheOnce {
        tape: DensityBody,
    },

    SplineGeneral {
        coordinate: DensityInstructionIndex,
        points: Vec<SplinePointInstruction>,
    },

    // this is not well optimized at this point, because both branches are always evaluated
    RangeChoice {
        input: DensityInstructionIndex,
        min_inclusive: f64,
        max_inclusive: f64,
        then: DensityInstructionIndex,
        otherwise: DensityInstructionIndex,
    },

    Clamped {
        input: DensityInstructionIndex,
        min_inclusive: f64,
        max_inclusive: f64,
    },

    Squeeze {
        argument: DensityInstructionIndex,
    },

    Result(DensityInstructionIndex),
}

#[derive(Debug)]
pub struct SplinePointInstruction {
    derivative: f64,
    location: f64,
    value: DensityInstructionIndex,
}

impl DensityBody {
    pub fn new(cell_size: (u8, u8, u8)) -> Self {
        DensityBody {
            cell_size,
            instructions: Vec::new(),
        }
    }

    pub fn add_instruction(&mut self, instr: DensityInstruction) -> DensityInstructionIndex {
        let idx = self.instructions.len();
        self.instructions.push(instr);
        idx as DensityInstructionIndex
    }
}
