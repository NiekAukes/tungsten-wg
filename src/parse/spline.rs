use crate::parse::{
    Density, Interned, MinecraftData,
    density::DensityParseFunctions,
    model::{DensityType, Spline, SplinePoint, SplineType, SplineValue},
};
use std::hash::{Hash, Hasher};

pub trait SplineParseFunctions<'m> {
    fn parse_spline(&self, value: &serde_json::Value) -> Spline<'m>;
}

impl<'m> SplineParseFunctions<'m> for MinecraftData<'m> {
    fn parse_spline(&self, value: &serde_json::Value) -> Spline<'m> {
        if let Some(obj) = value.as_object() {
            // parse coordinate as density function
            let coordinate = obj.get("coordinate").expect("expected coordinate");
            let coordinate = self.parse_density_function_from_value(coordinate);

            // parse spline points
            let points = obj
                .get("points")
                .expect("expected points")
                .as_array()
                .expect("expected points to be an array");
            let mut splinepoints = vec![];
            for p in points {
                let point_obj = p.as_object().expect("expected point to be an object");
                let derivative = point_obj
                    .get("derivative")
                    .expect("expected derivative")
                    .as_f64()
                    .expect("derivative must be a number");
                let location = point_obj
                    .get("location")
                    .expect("expected location")
                    .as_f64()
                    .expect("location must be a number");
                let value = point_obj.get("value").expect("expected value");
                let spline_value = if let Some(c) = value.as_f64() {
                    // value is a const
                    SplineValue::Const(c)
                } else {
                    // otherwise it must be another spline
                    let spline = self.parse_spline(value);
                    SplineValue::Spline(spline)
                };

                //let spline_value = self.arena.alloc(spline_value);
                splinepoints.push(SplinePoint {
                    derivative,
                    location,
                    value: spline_value,
                });
            }
            let spline = SplineType {
                coordinate: coordinate,
                spline_points: self.arena.alloc_slice_clone(&splinepoints),
            };
            self.arena.alloc(spline)
        } else {
            panic!("Spline must be an object: {:?}", value);
        }
    }
}
