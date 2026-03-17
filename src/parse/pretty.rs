use std::fmt;

use crate::parse::{
    model::DensityType,
    model::{SplinePoint, SplineType, SplineValue},
};

impl<'m> fmt::Display for DensityType<'m> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_indent(f, 0)
    }
}

impl<'m> DensityType<'m> {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = |n| " ".repeat(n);
        match self {
            DensityType::Const(val) => {
                writeln!(f, "{}Const({})", pad(indent), val)
            }
            DensityType::Noise {..} => {
                writeln!(f, "{}Noise(...)", pad(indent))
            }
            DensityType::Add { left, right } => {
                writeln!(f, "{}Add:", pad(indent))?;
                left.fmt_with_indent(f, indent + 2)?;
                right.fmt_with_indent(f, indent + 2)
            }
            DensityType::Multiply { left, right } => {
                writeln!(f, "{}Multiply:", pad(indent))?;
                left.fmt_with_indent(f, indent + 2)?;
                right.fmt_with_indent(f, indent + 2)
            }
            DensityType::Cache2d { argument }
            | DensityType::Squeeze { argument }
            | DensityType::Interpolated { argument }
            | DensityType::FlatCache { argument }
            | DensityType::ShiftA { argument }
            | DensityType::ShiftB { argument }
            | DensityType::CacheOnce { argument }
            | DensityType::Abs { argument }
            | DensityType::Square { argument }
            | DensityType::Cube { argument } => {
                writeln!(f, "{}{:?}:", pad(indent), self.variant_name())?;
                argument.fmt_with_indent(f, indent + 2)
            }
            DensityType::EndIslands => {
                writeln!(f, "{}EndIslands", pad(indent))
            }
            DensityType::YClampedGradient {
                from_y,
                to_y,
                from_value,
                to_value,
            } => {
                writeln!(
                    f,
                    "{}YClampedGradient(from_y={}, to_y={}, from_value={}, to_value={})",
                    pad(indent),
                    from_y,
                    to_y,
                    from_value,
                    to_value
                )
            }
            DensityType::OldBlendedNoise {
                smear_scale_multiplier,
                xz_factor,
                xz_scale,
                y_factor,
                y_scale,
            } => {
                writeln!(
                    f,
                    "{}OldBlendedNoise(smear={}, xz_factor={}, xz_scale={}, y_factor={}, y_scale={})",
                    pad(indent),
                    smear_scale_multiplier,
                    xz_factor,
                    xz_scale,
                    y_factor,
                    y_scale
                )
            }
            DensityType::ShiftedNoise {
                shift_y,
                xz_scale,
                y_scale,
                ..
            } => {
                writeln!(
                    f,
                    "{}ShiftedNoise(shift_y={}, xz_scale={}, y_scale={}, ...)",
                    pad(indent),
                    shift_y,
                    xz_scale,
                    y_scale
                )
            }
            DensityType::Spline { spline } => {
                writeln!(f, "{}Spline({})", pad(indent), spline)
            }
            DensityType::Min { left, right } | DensityType::Max { left, right } => {
                writeln!(f, "{}{:?}:", pad(indent), self.variant_name())?;
                left.fmt_with_indent(f, indent + 2)?;
                right.fmt_with_indent(f, indent + 2)
            }
            DensityType::RangeChoice {
                input,
                min_inclusive,
                max_exclusive,
                when_in_range,
                when_out_of_range,
            } => {
                writeln!(
                    f,
                    "{}RangeChoice(min={}, max={}):",
                    pad(indent),
                    min_inclusive,
                    max_exclusive
                )?;
                writeln!(f, "{}Input:", pad(indent + 2))?;
                input.fmt_with_indent(f, indent + 4)?;
                writeln!(f, "{}WhenInRange:", pad(indent + 2))?;
                when_in_range.fmt_with_indent(f, indent + 4)?;
                writeln!(f, "{}WhenOutOfRange:", pad(indent + 2))?;
                when_out_of_range.fmt_with_indent(f, indent + 4)
            }
            DensityType::Clamp { input, min, max } => {
                writeln!(f, "{}Clamp(min={}, max={}):", pad(indent), min, max)?;
                input.fmt_with_indent(f, indent + 2)
            }
            DensityType::WeirdScaledSampler {
                rarity_value_mapper,
                ..
            } => {
                writeln!(
                    f,
                    "{}WeirdScaledSampler(mapper={})",
                    pad(indent),
                    rarity_value_mapper
                )
            }
            DensityType::NamedDensityReference { name, argument } => {
                writeln!(f, "{}NamedDensityReference(name={}):", pad(indent), name)?;
                argument.fmt_with_indent(f, indent + 2)
            }
        }
    }

    fn variant_name(&self) -> &'static str {
        match self {
            DensityType::Cache2d { .. } => "Cache2d",
            DensityType::Squeeze { .. } => "Squeeze",
            DensityType::Interpolated { .. } => "Interpolated",
            DensityType::FlatCache { .. } => "FlatCache",
            DensityType::ShiftA { .. } => "ShiftA",
            DensityType::ShiftB { .. } => "ShiftB",
            DensityType::CacheOnce { .. } => "CacheOnce",
            DensityType::Abs { .. } => "Abs",
            DensityType::Square { .. } => "Square",
            DensityType::Cube { .. } => "Cube",
            DensityType::Min { .. } => "Min",
            DensityType::Max { .. } => "Max",
            _ => "Unknown",
        }
    }
}

impl<'m> fmt::Display for SplineType<'m> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_indent(f, 0)
    }
}

impl<'m> SplineType<'m> {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = |n| " ".repeat(n);
        writeln!(f, "{}SplineType:", pad(indent))?;
        writeln!(f, "{}Coordinate:", pad(indent + 2))?;
        self.coordinate.fmt_with_indent(f, indent + 4)?;
        writeln!(f, "{}Points:", pad(indent + 2))?;
        for point in self.spline_points.iter() {
            point.fmt_with_indent(f, indent + 4)?;
        }
        Ok(())
    }
}

impl<'m> SplinePoint<'m> {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = |n| " ".repeat(n);
        writeln!(
            f,
            "{}Point(location={}, derivative={}):",
            pad(indent),
            self.location,
            self.derivative
        )?;
        self.value.fmt_with_indent(f, indent + 2)
    }
}

impl<'m> SplineValue<'m> {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = |n| " ".repeat(n);
        match self {
            SplineValue::Const(val) => writeln!(f, "{}Const({})", pad(indent), val),
            SplineValue::Spline(spline) => {
                writeln!(f, "{}Spline:", pad(indent))?;
                spline.fmt_with_indent(f, indent + 2)
            }
        }
    }
}
