use std::collections::{HashMap, HashSet};

use crate::{
    orchestrate::model::{Orchestration, ShaderDependency, ShaderRef},
    spmt::{
        model::Addr,
        pretty::{PrettyPrint, Printer},
    },
};

impl<'a> PrettyPrint for Orchestration<'a> {
    fn pretty(&self, p: &mut Printer) {
        // let mut visited = HashSet::new();

        // p.line("digraph Shaders {");
        // p.indent();
        // p.line("node [shape=box];");

        // for shader in &self.shaders {
        //     self.visit(shader, p, &mut visited);
        // }

        // p.dedent();
        // p.line("}");
    }
}

impl<'a> Orchestration<'a> {
    fn visit(
        &self,
        shaderdep: &ShaderDependency<'a>,
        p: &mut Printer,
        visited: &mut HashSet<*const ()>,
    ) {
        let addr = shaderdep.shader.addr();
        if !visited.insert(addr) {
            return;
        }

        // Stable DOT node id (pointer-based but cached)
        let node_id = p.anon_name(shaderdep.shader, "shader");

        // Emit node
        p.line(&format!(
            r#"{id} [label="{label}"];"#,
            id = node_id,
            label = shaderdep.shader.name
        ));

        // Emit edges
        for input in &shaderdep.shader.inputs {
            let child_id = p.anon_name(input, "shader");

            p.line(&format!(
                r#"{from} -> {to};"#,
                from = node_id,
                to = child_id
            ));

            self.visit(input, p, visited);
        }
    }
}

pub fn pretty_dependency_map<'m>(
    deps: &HashMap<ShaderRef<'m>, Vec<ShaderRef<'m>>>,
    p: &mut Printer,
) {
    let mut emitted = HashSet::new();

    p.line("digraph ShaderDependencies {");
    p.indent();
    p.line("node [shape=box];");

    // Emit all nodes
    for (shader, dependents) in deps {
        let id = p.anon_name(*shader, "shader");

        if emitted.insert(shader.addr()) {
            p.line(&format!(
                r#"{id} [label="{label}"];"#,
                id = id,
                label = shader.name
            ));
        }

        for dep in dependents {
            let dep_id = p.anon_name(*dep, "shader");

            if emitted.insert(dep.addr()) {
                p.line(&format!(
                    r#"{id} [label="{label}"];"#,
                    id = dep_id,
                    label = dep.name
                ));
            }
        }
    }

    // Emit edges
    for (shader, dependents) in deps {
        let from_id = p.anon_name(*shader, "shader");

        for dep in dependents {
            let to_id = p.anon_name(*dep, "shader");

            p.line(&format!(r#"{from} -> {to};"#, from = from_id, to = to_id));
        }
    }

    p.dedent();
    p.line("}");
}
impl<'m> Orchestration<'m> {
    pub fn pretty_wave_graph(&self, p: &mut Printer) {
        let waves = self.arrange_waves();

        let mut wave_of: HashMap<*const (), usize> = HashMap::new();

        for (i, wave) in waves.iter().enumerate() {
            for dep in wave {
                wave_of.insert(dep.shader.addr(), i);
            }
        }

        let mut wave_edges = HashSet::<(usize, usize)>::new();

        for wave in &waves {
            for dep in wave {
                let from_wave = wave_of[&dep.shader.addr()];

                for input_dep in &dep.shader.inputs {
                    let to_wave = wave_of[&input_dep.shader.addr()];

                    if from_wave != to_wave {
                        wave_edges.insert((to_wave, from_wave));
                    }
                }
            }
        }

        p.line("digraph WaveGraph {");
        p.indent();
        p.line("rankdir=LR;");
        p.line("node [shape=box, style=filled, fillcolor=lightgray];");

        for (i, wave) in waves.iter().enumerate() {
            p.line(&format!(
                r#"wave{} [label="Wave {}\n{} shaders"];"#,
                i,
                i,
                wave.len()
            ));
        }

        for (from, to) in wave_edges {
            p.line(&format!("wave{} -> wave{};", from, to));
        }

        p.dedent();
        p.line("}");
    }

    pub fn pretty_wave_dependencies(&self, p: &mut Printer) {
        let waves = self.arrange_waves();

        let mut wave_of: HashMap<ShaderDependency<'_>, usize> = HashMap::new();

        for (i, wave) in waves.iter().enumerate() {
            for dep in wave {
                wave_of.insert(dep.clone(), i);
            }
        }

        p.line("digraph ShaderWaves {");
        p.indent();
        p.line("rankdir=LR;");
        p.line("node [shape=box];");

        // Emit wave clusters
        for (i, wave) in waves.iter().enumerate() {
            p.line(&format!("subgraph cluster_wave_{} {{", i));
            p.indent();
            p.line(&format!(r#"label="wave {}";"#, i));
            p.line("rank=same;");

            for dep in wave {
                let id = dep.get_id();
                let dimensions = dep.dimensions;
                let scale = dep.scaled_origin.as_float();
                p.line(&format!(
                    r#"{id} [label="{label}\nDimensions: ({x} {y} {z})\nScale: ({sx} {sy} {sz})"];"#,
                    id = id,
                    label = dep.shader.name,
                    x = dimensions.0,
                    y = dimensions.1,
                    z = dimensions.2,
                    sx = scale.0,
                    sy = scale.1,
                    sz = scale.2,
                ));
            }

            p.dedent();
            p.line("}");
        }

        // Emit edges
        for wave in &waves {
            for dep in wave {
                let from_id = dep.get_id();
                let from_wave = wave_of[dep];

                for input_dep in &dep.shader.inputs {
                    let to_id = input_dep.get_id();
                    let to_wave = wave_of[input_dep];

                    let distance = from_wave as isize - to_wave as isize;

                    if distance > 1 {
                        p.line(&format!(
                            r#"{from} -> {to} [style=dashed, color=gray];"#,
                            from = from_id,
                            to = to_id
                        ));
                    } else {
                        p.line(&format!(r#"{from} -> {to};"#, from = from_id, to = to_id));
                    }
                }
            }
        }

        p.dedent();
        p.line("}");
    }
}

impl ShaderDependency<'_> {
    pub fn get_id(&self) -> String {
        format!(
            "<shader_{:?}_{}x{}x{}_{}_{}_{}>",
            self.shader.addr(),
            self.dimensions.0,
            self.dimensions.1,
            self.dimensions.2,
            self.scaled_origin.x,
            self.scaled_origin.y,
            self.scaled_origin.z
        )
    }
}
