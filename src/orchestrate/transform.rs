use std::collections::{HashMap, HashSet};

use crate::orchestrate::dot::pretty_dependency_map;
use crate::orchestrate::model::{Shader, ShaderRef};
use crate::spmt::model::{Addr, DensityFunctionRef};
use crate::spmt::pretty::Printer;
use crate::{orchestrate::model::Orchestration, spmt::model::SPMT};

pub struct Transformer<'a, 'm> {
    arena: &'m bumpalo::Bump,
    orchestration: Orchestration<'m>,
    cache: HashMap<DensityFunctionRef<'a>, ShaderRef<'m>>,
}

pub fn transform_from_spmt<'a, 'm>(spmt: SPMT<'a>, arena: &'m bumpalo::Bump) -> Orchestration<'m> {
    //let arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024); // 1 MB initial capacity
    let orchestration = Orchestration::new(&arena);

    let mut transformer = Transformer::new(&arena, orchestration);
    for density_function in spmt.density_functions {
        transformer.transform_density_function_to_shader(density_function);
    }
    let orchestration = transformer.orchestration;

    orchestration
}

impl<'m, 'a> Transformer<'a, 'm> {
    pub fn new(arena: &'m bumpalo::Bump, orchestration: Orchestration<'m>) -> Self {
        Self {
            arena,
            orchestration,
            cache: HashMap::new(),
        }
    }

    fn transform_density_function_to_shader(
        &mut self,
        density_function: DensityFunctionRef<'a>,
    ) -> ShaderRef<'m> {
        if let Some(shader_ref) = self.cache.get(&density_function) {
            return *shader_ref;
        }
        let name = density_function
            .canonical_name
            .clone()
            .unwrap_or_else(|| "unnamed".to_string())
            .clone();
        let source = format!("density_function: {}", name);
        let mut dependencies = vec![];

        for input in &density_function.density_inputs {
            let shader_ref = self.transform_density_function_to_shader(input.density_function);
            dependencies.push(shader_ref);
        }

        let shader = Shader {
            name: name.clone(),
            source,
            inputs: dependencies,
        };
        let shader_ref = self.orchestration.add_shader(shader);
        self.cache.insert(density_function, shader_ref);
        shader_ref
    }
}

impl<'m> Orchestration<'m> {
    pub fn arrange_waves(&self) -> Vec<Vec<ShaderRef<'m>>> {
        // rearrange shaders based on their dependencies
        // ideally, you want orchestration to have as few bottlenecks as possible,
        // or as many parallelizable shaders as possible, so you want to arrange the shaders in a way that minimizes the number of shaders that depend on each other

        // we can achieve this with a wave-like algorithm, where we start with shaders that have no dependencies,
        // count the dependencies of each shader, and then iteratively add shaders with the least dependencies until we have arranged all shaders

        let mut arranged: Vec<Vec<ShaderRef<'m>>> = Vec::new();
        let mut dependencies: HashMap<ShaderRef<'m>, Vec<ShaderRef<'m>>> = HashMap::new();
        for shader in &self.shaders {
            dependencies.entry(*shader).or_default();
            for input in &shader.inputs {
                dependencies.entry(*input).or_default().push(*shader);
            }
        }

        while dependencies.len() > 0 {
            let mut wave: Vec<ShaderRef<'m>> = Vec::new();
            for (shader, deps) in &dependencies {
                if deps.len() == 0 {
                    wave.push(*shader);
                }
            }
            if wave.len() == 0 {
                let mut printer = Printer::new();
                pretty_dependency_map(&dependencies, &mut printer);
                let orchestration_output = printer.finish();
                std::fs::write("orchestration.dot", orchestration_output)
                    .expect("Unable to write file");
                panic!("Cyclic dependency detected");
            }
            for shader in &wave {
                dependencies.remove(&shader);
                for dependent in dependencies.values_mut() {
                    dependent.retain(|&s| s != *shader);
                }
            }
            arranged.push(wave);
        }
        arranged.reverse(); // reverse to get correct order (shaders with no dependencies should be first)
        arranged
    }

    pub fn pretty_waves(&self, p: &mut Printer) {
        let waves = self.arrange_waves();
        let mut emitted = HashSet::new();

        p.line("digraph ShaderWaves {");
        p.indent();
        p.line("node [shape=box];");

        // Emit all nodes (grouped by wave)
        for (i, wave) in waves.iter().enumerate() {
            p.line(&format!("subgraph cluster_wave_{} {{", i));
            p.indent();
            p.line(r#"label="wave";"#);
            p.line("rank=same;");

            for shader in wave {
                let node_id = p.anon_name(*shader, "shader");

                if emitted.insert((*shader).addr()) {
                    p.line(&format!(
                        r#"{id} [label="{label}"];"#,
                        id = node_id,
                        label = shader.name
                    ));
                } else {
                    // still need to mention node in rank group
                    p.line(&node_id);
                }
            }

            p.dedent();
            p.line("}");
        }

        // Emit dependency edges
        for shader in &self.shaders {
            let from_id = p.anon_name(*shader, "shader");

            for input in &shader.inputs {
                let to_id = p.anon_name(*input, "shader");

                p.line(&format!(r#"{from} -> {to};"#, from = from_id, to = to_id));
            }
        }

        p.dedent();
        p.line("}");
    }
}
