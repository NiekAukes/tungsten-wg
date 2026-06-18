// transform_spmt/spline.rs

pub mod old;
pub mod spline_new;

use serde::de::value;

use crate::parse::model::Density;
use crate::spmt::model::DensityInput;
use crate::transform_spmt::density::{DensityBuilder, make_rpos3};
use crate::transform_spmt::{newvar, prefixvar};
use crate::{
    parse::model::{Spline, SplinePoint, SplineValue},
    spmt::model::{
        BinaryOperator, Expression, Function, FunctionRef, Statement, Var, Variable, VariableType,
    },
};

/* New spline idea:

Make a decision tree of the spline points
decision_tree: [(u8, u24, f32)] = [(next_coord_idx, next_decision_idx, location)]
values: [f32] = [value0, value1, value2, ...] (the values of the spline points in the same order as the decision tree)

coord = input[0]
decision_idx = 0
decision = decision_tree[decision_idx]
value = 0.0
prev_value = 0.0
derivative = 0.0
location = 0.0

next_coord_idx, next_decision_idx, location = decision

for _ in max_depth {
    decision_idx = next_decision_idx * (coord >= location) + (coord < location)
    if coord < location && (nc > max_coord_idx || decision_idx > max_decision_idx) {
        value, derivative = values[decision_idx - max_decision_idx]
        prev_value = values[decision_idx - max_decision_idx - 1]
        break
    }

    decision = decision_tree[decision_idx]
    next_coord_idx, next_decision_idx, location = decision
    coord = input[next_coord_idx]
}

// get the value
// interpolate or extrapolate based on the last decision

// extrapolation: return (((coordinate - 1f) * 0.38940096f) + 0.69000006f);
// hermite: fn hermite(t: f32, p0_: f32, p1_: f32, m0_: f32, m1_: f32) -> f32 {
    let t2_ = (t * t);
    let t3_ = (t2_ * t);
    return (((((((2f * t3_) - (3f * t2_)) + 1f) * p0_) + (((t3_ - (2f * t2_)) + t) * m0_)) + (((-2f * t3_) + (3f * t2_)) * p1_)) + ((t3_ - t2_) * m1_));
}

if nc > max_coord_idx {
    return (((coordinate - location) * derivative) + value);
} else {
    return hermite(t, prev_value, value, derivative, next_derivative);
}


*/

impl<'a, 'm> DensityBuilder<'a, 'm> {
    /// Main entry point for spline lowering - routes to old or new implementation based on settings
    pub fn lower_spline_definition(
        &mut self,
        spline: Spline<'a>,
        canonical_name: Option<String>,
    ) -> Expression<'m> {
        if self.builder_settings.use_new_spline {
            self.lower_spline_definition_new(spline, canonical_name)
        } else {
            self.lower_spline_definition_old(spline, canonical_name)
        }
    }
}
