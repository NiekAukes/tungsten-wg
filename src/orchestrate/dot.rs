use std::collections::{HashMap, HashSet};

use crate::{
    orchestrate::model::{Orchestration, ShaderRef},
    spmt::{
        model::Addr,
        pretty::{PrettyPrint, Printer},
    },
};

impl<'a> PrettyPrint for Orchestration<'a> {
    fn pretty(&self, p: &mut Printer) {
        let mut visited = HashSet::new();

        p.line("digraph Shaders {");
        p.indent();
        p.line("node [shape=box];");

        for shader in &self.shaders {
            self.visit(shader, p, &mut visited);
        }

        p.dedent();
        p.line("}");
    }
}

impl<'a> Orchestration<'a> {
    fn visit(&self, shader: &ShaderRef<'a>, p: &mut Printer, visited: &mut HashSet<*const ()>) {
        let addr = (*shader).addr();
        if !visited.insert(addr) {
            return;
        }

        // Stable DOT node id (pointer-based but cached)
        let node_id = p.anon_name(*shader, "shader");

        // Emit node
        p.line(&format!(
            r#"{id} [label="{label}"];"#,
            id = node_id,
            label = shader.name
        ));

        // Emit edges
        for input in &shader.inputs {
            let child_id = p.anon_name(*input, "shader");

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
        let mut waves = self.arrange_waves();

        // Map shader -> wave index
        let mut wave_of = HashMap::new();
        for (i, wave) in waves.iter().enumerate() {
            for shader in wave {
                wave_of.insert(*shader, i);
            }
        }

        // Collect wave-to-wave edges
        let mut wave_edges = HashSet::<(usize, usize)>::new();

        for shader in &self.shaders {
            let from_wave = wave_of[shader];

            for input in &shader.inputs {
                let to_wave = wave_of[input];

                if from_wave != to_wave {
                    wave_edges.insert((from_wave, to_wave));
                }
            }
        }

        // Emit DOT
        p.line("digraph WaveGraph {");
        p.indent();
        p.line("rankdir=LR;");
        p.line("node [shape=box, style=filled, fillcolor=lightgray];");

        // Emit wave nodes
        for (i, wave) in waves.iter().enumerate() {
            p.line(&format!(
                r#"wave{} [label="Wave {}\n{} shaders"];"#,
                i,
                i,
                wave.len()
            ));
        }

        // Emit aggregated wave edges
        for (from, to) in wave_edges {
            p.line(&format!("wave{} -> wave{};", to, from));
        }

        p.dedent();
        p.line("}");
    }

    pub fn pretty_wave_dependencies(&self, p: &mut Printer) {
        let waves = self.arrange_waves();

        // Map shader -> wave index
        let mut wave_of: HashMap<ShaderRef<'m>, usize> = HashMap::new();
        for (i, wave) in waves.iter().enumerate() {
            for shader in wave {
                wave_of.insert(*shader, i);
            }
        }

        p.line("digraph ShaderWaves {");
        p.indent();

        p.line("rankdir=LR;");
        p.line("node [shape=box];");

        // 1️⃣ Emit wave groups
        for (i, wave) in waves.iter().enumerate() {
            p.line(&format!("subgraph cluster_wave_{} {{", i));
            p.indent();

            p.line(&format!(r#"label="wave {}";"#, i));
            p.line("rank=same;");

            for shader in wave {
                let id = p.anon_name(*shader, "shader");
                p.line(&format!(
                    r#"{id} [label="{label}"];"#,
                    id = id,
                    label = shader.name
                ));
            }

            p.dedent();
            p.line("}");
        }

        // 2️⃣ Emit edges (preserve real dependencies)
        for shader in &self.shaders {
            let from_id = p.anon_name(*shader, "shader");
            let from_wave = wave_of.get(shader).copied().unwrap_or(0);

            for input in &shader.inputs {
                let to_id = p.anon_name(*input, "shader");
                let to_wave = wave_of.get(input).copied().unwrap_or(0);

                let wave_distance = from_wave as isize - to_wave as isize;

                if wave_distance > 1 {
                    // Long jump edge
                    p.line(&format!(
                        r#"{from} -> {to} [style=dashed, color=gray, label="Δ{}"];"#,
                        wave_distance,
                        from = from_id,
                        to = to_id
                    ));
                } else {
                    // Adjacent wave edge
                    p.line(&format!(r#"{from} -> {to};"#, from = from_id, to = to_id));
                }
            }
        }

        p.dedent();
        p.line("}");
    }
}
