use std::collections::HashMap;

use crate::parse::model::{Density, DensityType, Spline, SplineValue};

/// Emits a DOT digraph for a density function tree/DAG.
///
/// Shared (interned) nodes are deduplicated by pointer identity, so the output
/// is a proper DAG rather than a tree.
pub struct DotPrinter {
    nodes: Vec<String>,
    edges: Vec<String>,
    density_ids: HashMap<*const (), String>,
    spline_ids: HashMap<*const (), String>,
    counter: usize,
}

impl DotPrinter {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            density_ids: HashMap::new(),
            spline_ids: HashMap::new(),
            counter: 0,
        }
    }

    fn next_id(&mut self) -> String {
        let id = format!("n{}", self.counter);
        self.counter += 1;
        id
    }

    /// Escape a string so it is safe to embed in a DOT double-quoted label.
    fn escape(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }

    /// Visit a density node. Returns the stable DOT node id for this node.
    /// Emits a node declaration on first visit; subsequent visits just return
    /// the cached id so edges can point to the existing node.
    pub fn visit_density<'m>(&mut self, density: Density<'m>) -> String {
        let ptr = density as *const _ as *const ();
        if let Some(id) = self.density_ids.get(&ptr) {
            return id.clone();
        }

        let id = self.next_id();
        self.density_ids.insert(ptr, id.clone());

        let label = Self::label_for(density);
        self.nodes
            .push(format!("    {} [label=\"{}\"];", id, Self::escape(&label)));

        // Clone id before the mutable borrow in visit_density_children
        let id_owned = id.clone();
        self.visit_density_children(&id_owned, density);

        id
    }

    fn label_for(density: &DensityType<'_>) -> String {
        match density {
            DensityType::Const(v) => {
                // format only the first few decimal places
                // but don't force it to always have a decimal point (e.g. "1" instead of "1.00")
                let a = format!("{:.4}", v);
                let a = a.trim_end_matches('0').trim_end_matches('.').to_string();
                format!("Const({})", a)
            }
            DensityType::Noise {
                name,
                xz_scale,
                y_scale,
                ..
            } => {
                format!("Noise\n{}\nxz={} y={}", name, xz_scale, y_scale)
            }
            DensityType::Add { .. } => "Add".to_string(),
            DensityType::Multiply { .. } => "Multiply".to_string(),
            DensityType::Cache2d { .. } => "Cache2d".to_string(),
            DensityType::Squeeze { .. } => "Squeeze".to_string(),
            DensityType::Interpolated { .. } => "Interpolated".to_string(),
            DensityType::FlatCache { .. } => "FlatCache".to_string(),
            DensityType::CacheOnce { .. } => "CacheOnce".to_string(),
            DensityType::Abs { .. } => "Abs".to_string(),
            DensityType::Square { .. } => "Square".to_string(),
            DensityType::Cube { .. } => "Cube".to_string(),
            DensityType::Min { .. } => "Min".to_string(),
            DensityType::Max { .. } => "Max".to_string(),
            DensityType::EndIslands => "EndIslands".to_string(),
            DensityType::YClampedGradient {
                from_y,
                to_y,
                from_value,
                to_value,
            } => {
                format!(
                    "YClampedGradient\nfrom_y={} to_y={}\nfrom_v={} to_v={}",
                    from_y, to_y, from_value, to_value
                )
            }
            DensityType::OldBlendedNoise {
                xz_scale, y_scale, ..
            } => {
                format!("OldBlendedNoise\nxz={} y={}", xz_scale, y_scale)
            }
            DensityType::ShiftedNoise {
                name,
                xz_scale,
                y_scale,
                ..
            } => {
                format!("ShiftedNoise\n{}\nxz={} y={}", name, xz_scale, y_scale)
            }
            DensityType::ShiftA { name, .. } => format!("ShiftA\n{}", name),
            DensityType::ShiftB { name, .. } => format!("ShiftB\n{}", name),
            DensityType::Spline { .. } => "Spline".to_string(),
            DensityType::RangeChoice {
                min_inclusive,
                max_exclusive,
                ..
            } => {
                format!("RangeChoice\nmin={} max={}", min_inclusive, max_exclusive)
            }
            DensityType::XNegative {
                neg_x_multiplier, ..
            } => {
                format!("XNegative(mul={})", neg_x_multiplier)
            }
            DensityType::Clamp { min, max, .. } => format!("Clamp\nmin={} max={}", min, max),
            DensityType::WeirdScaledSampler {
                noise_name,
                rarity_value_mapper,
                ..
            } => {
                format!(
                    "WeirdScaledSampler\n{}\n{}",
                    noise_name, rarity_value_mapper
                )
            }
            DensityType::NamedDensityReference { name, .. } => {
                format!("NamedDensityRef\n{}", name)
            }
        }
    }

    fn visit_density_children<'m>(&mut self, parent_id: &str, density: Density<'m>) {
        match density {
            DensityType::Add { left, right }
            | DensityType::Multiply { left, right }
            | DensityType::Min { left, right }
            | DensityType::Max { left, right } => {
                let left_id = self.visit_density(left);
                let right_id = self.visit_density(right);
                self.edges.push(format!(
                    "    {} -> {} [label=\"left\"];",
                    parent_id, left_id
                ));
                self.edges.push(format!(
                    "    {} -> {} [label=\"right\"];",
                    parent_id, right_id
                ));
            }

            DensityType::Cache2d { argument }
            | DensityType::Squeeze { argument }
            | DensityType::Interpolated { argument }
            | DensityType::FlatCache { argument }
            | DensityType::CacheOnce { argument }
            | DensityType::Abs { argument }
            | DensityType::Square { argument }
            | DensityType::Cube { argument } => {
                let child_id = self.visit_density(argument);
                self.edges
                    .push(format!("    {} -> {};", parent_id, child_id));
            }

            DensityType::ShiftedNoise {
                shift_x,
                shift_y,
                shift_z,
                ..
            } => {
                let x_id = self.visit_density(shift_x);
                let y_id = self.visit_density(shift_y);
                let z_id = self.visit_density(shift_z);
                self.edges.push(format!(
                    "    {} -> {} [label=\"shift_x\"];",
                    parent_id, x_id
                ));
                self.edges.push(format!(
                    "    {} -> {} [label=\"shift_y\"];",
                    parent_id, y_id
                ));
                self.edges.push(format!(
                    "    {} -> {} [label=\"shift_z\"];",
                    parent_id, z_id
                ));
            }

            DensityType::Spline { spline } => {
                let coord_id = self.visit_density(spline.coordinate);
                self.edges.push(format!(
                    "    {} -> {} [label=\"coord\"];",
                    parent_id, coord_id
                ));
                let parent_owned = parent_id.to_string();
                for point in spline.spline_points.iter() {
                    if let SplineValue::Spline(inner) = &point.value {
                        let inner_id = self.visit_inner_spline(*inner);
                        self.edges.push(format!(
                            "    {} -> {} [label=\"pt@{:.2}\"];",
                            parent_owned, inner_id, point.location
                        ));
                    }
                }
            }

            DensityType::RangeChoice {
                input,
                when_in_range,
                when_out_of_range,
                ..
            } => {
                let in_id = self.visit_density(input);
                let wir_id = self.visit_density(when_in_range);
                let woor_id = self.visit_density(when_out_of_range);
                self.edges
                    .push(format!("    {} -> {} [label=\"input\"];", parent_id, in_id));
                self.edges.push(format!(
                    "    {} -> {} [label=\"in_range\"];",
                    parent_id, wir_id
                ));
                self.edges.push(format!(
                    "    {} -> {} [label=\"out_range\"];",
                    parent_id, woor_id
                ));
            }

            DensityType::XNegative { argument, .. }
            | DensityType::NamedDensityReference { argument, .. } => {
                let child_id = self.visit_density(argument);
                self.edges
                    .push(format!("    {} -> {};", parent_id, child_id));
            }

            DensityType::Clamp { input, .. } | DensityType::WeirdScaledSampler { input, .. } => {
                let child_id = self.visit_density(input);
                self.edges
                    .push(format!("    {} -> {};", parent_id, child_id));
            }

            // Leaf nodes – no density children
            DensityType::Const(_)
            | DensityType::Noise { .. }
            | DensityType::EndIslands
            | DensityType::YClampedGradient { .. }
            | DensityType::OldBlendedNoise { .. }
            | DensityType::ShiftA { .. }
            | DensityType::ShiftB { .. } => {}
        }
    }

    /// Visit a nested SplineType (not a DensityType::Spline wrapper).
    /// These are emitted as diamond-shaped nodes to distinguish them visually.
    fn visit_inner_spline<'m>(&mut self, spline: Spline<'m>) -> String {
        let ptr = spline as *const _ as *const ();
        if let Some(id) = self.spline_ids.get(&ptr) {
            return id.clone();
        }

        let id = self.next_id();
        self.spline_ids.insert(ptr, id.clone());

        self.nodes
            .push(format!("    {} [label=\"Spline\", shape=diamond];", id));

        let coord_id = self.visit_density(spline.coordinate);
        self.edges
            .push(format!("    {} -> {} [label=\"coord\"];", id, coord_id));

        let id_owned = id.clone();
        for point in spline.spline_points.iter() {
            if let SplineValue::Spline(inner) = &point.value {
                let inner_id = self.visit_inner_spline(*inner);
                self.edges.push(format!(
                    "    {} -> {} [label=\"pt@{:.2}\"];",
                    id_owned, inner_id, point.location
                ));
            }
        }

        id
    }

    /// Consume the printer and return the complete DOT source.
    pub fn finish(self) -> String {
        let mut out = String::new();
        out.push_str("digraph DensityFunction {\n");
        out.push_str("    node [shape=box];\n");
        for node in &self.nodes {
            out.push_str(node);
            out.push('\n');
        }
        for edge in &self.edges {
            out.push_str(edge);
            out.push('\n');
        }
        out.push('}');
        out
    }
}

/// Convenience function: produce a DOT string for a single density root.
pub fn print_density_dot<'m>(density: Density<'m>) -> String {
    let mut printer = DotPrinter::new();
    printer.visit_density(density);
    printer.finish()
}
