use crate::{
    parse::model::{Density, DensityType, NormalNoise},
    tape::{DensityBody, DensityInstruction, DensityInstructionIndex},
};

pub struct DensityFunctionLowerer {
    body: DensityBody,
}

impl DensityFunctionLowerer {
    pub fn new(cell_size: (u8, u8, u8)) -> Self {
        Self {
            body: DensityBody {
                cell_size,
                instructions: Vec::new(),
            },
        }
    }

    pub fn into_body(self) -> DensityBody {
        self.body
    }

    pub fn mark_result(&mut self, idx: DensityInstructionIndex) {
        self.body.add_instruction(DensityInstruction::Result(idx));
    }

    pub fn lower(&mut self, df: &Density<'_>) -> DensityInstructionIndex {
        match *df {
            DensityType::Const(c) => {
                let instr: DensityInstruction = DensityInstruction::Constant(*c);
                self.body.add_instruction(instr)
            }
            DensityType::Noise(normal_noise_type) => self.lower_noise(normal_noise_type),
            DensityType::Add { left, right } => {
                let left_idx = self.lower(left);
                let right_idx = self.lower(right);
                let instr = DensityInstruction::Add(left_idx, right_idx);
                self.body.add_instruction(instr)
            }
            DensityType::Multiply { left, right } => {
                let left_idx = self.lower(left);
                let right_idx = self.lower(right);
                let instr = DensityInstruction::Multiply(left_idx, right_idx);
                self.body.add_instruction(instr)
            }
            DensityType::Cache2d { argument } => {
                let mut b = DensityFunctionLowerer::new((
                    self.body.cell_size.0,
                    255,
                    self.body.cell_size.2,
                ));
                let res = b.lower(argument);
                b.mark_result(res);
                let body = b.into_body();
                let instr = DensityInstruction::Cache2D { tape: body };
                self.body.add_instruction(instr)
            }
            DensityType::Squeeze { argument } => {
                let instr = DensityInstruction::Squeeze {
                    argument: self.lower(argument),
                };
                self.body.add_instruction(instr)
            }
            DensityType::Interpolated { argument } => {
                let mut b = DensityFunctionLowerer::new((4, 8, 4)); // fixed 4x8x4 cache for interpolated
                let res = b.lower(argument);
                b.mark_result(res);
                let body = b.into_body();
                let instr = DensityInstruction::Cache2D { tape: body };
                self.body.add_instruction(instr)
            }
            DensityType::EndIslands => todo!(),
            DensityType::YClampedGradient {
                from_y,
                to_y,
                from_value,
                to_value,
            } => todo!(),
            DensityType::FlatCache { argument } => todo!(),
            DensityType::OldBlendedNoise {
                smear_scale_multiplier,
                xz_factor,
                xz_scale,
                y_factor,
                y_scale,
            } => todo!(),
            DensityType::ShiftedNoise {
                noise,
                shift_x,
                shift_y,
                shift_z,
                xz_scale,
                y_scale,
            } => todo!(),
            DensityType::ShiftA { argument } => todo!(),
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
        }
    }

    pub fn lower_noise(&mut self, normal_noise_type: NormalNoise) -> DensityInstructionIndex {
        if normal_noise_type.amplitudes.len() <= 4 {
            let instr = DensityInstruction::Noise4 {
                first_octave: normal_noise_type.first_octave,
                amplitudes: [
                    *normal_noise_type.amplitudes.get(0).unwrap_or(&0.0),
                    *normal_noise_type.amplitudes.get(1).unwrap_or(&0.0),
                    *normal_noise_type.amplitudes.get(2).unwrap_or(&0.0),
                    *normal_noise_type.amplitudes.get(3).unwrap_or(&0.0),
                ],
                length: normal_noise_type.amplitudes.len() as u8,
            };
            self.body.add_instruction(instr)
        } else {
            let instr = DensityInstruction::NoiseGeneral {
                first_octave: normal_noise_type.first_octave,
                amplitudes: normal_noise_type.amplitudes.to_vec(),
            };
            self.body.add_instruction(instr)
        }
    }
}
