use std::collections::{BTreeSet, HashMap, HashSet};

use crate::orchestrate::Scale;
use crate::orchestrate::dot::pretty_dependency_map;
use crate::orchestrate::model::{Shader, ShaderDependency, ShaderRef};
use crate::spmt::model::{Addr, DensityFunctionRef};
use crate::spmt::pretty::Printer;
use crate::{orchestrate::model::Orchestration, spmt::model::SPMT};

pub struct Transformer<'a, 'm> {
    arena: &'m bumpalo::Bump,
    orchestration: Orchestration<'m>,
    cache: HashMap<DensityFunctionRef<'a>, ShaderRef<'m>>,
}

pub fn transform_from_spmt<'a, 'm>(spmt: &SPMT<'a>, arena: &'m bumpalo::Bump) -> Orchestration<'m> {
    //let arena = bumpalo::Bump::with_capacity(1 * 1024 * 1024); // 1 MB initial capacity
    let orchestration = Orchestration::new(&arena);

    let mut transformer = Transformer::new(&arena, orchestration);
    for density_function in &spmt.density_functions {
        transformer.transform_density_function_to_shader(*density_function, (16, 256, 16));
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
        dimensions: (i32, i32, i32),
    ) -> ShaderRef<'m> {
        if let Some(shader_ref) = self.cache.get(&density_function) {
            return *shader_ref;
        }
        let name = density_function.canonical_name.as_ref().unwrap();
        let source = format!("density_function: {}", name);
        let mut dependencies = vec![];

        for input in &density_function.density_inputs {
            let shader_ref =
                self.transform_density_function_to_shader(input.density_function, input.dimensions);
            dependencies.push(ShaderDependency {
                shader: shader_ref,
                scaled_origin: Scale::new(
                    input.scaled_origin.0,
                    input.scaled_origin.1,
                    input.scaled_origin.2,
                ),
                dimensions: input.dimensions,
            });
        }

        let shader = Shader {
            name: name.clone(),
            source,
            inputs: dependencies,
            permutation_tables: density_function.permutation_table_inputs.clone(),
        };
        let shader_ref = self.orchestration.add_main_shader(shader, dimensions);
        self.cache.insert(density_function, shader_ref);
        shader_ref
    }
}

impl<'m> Orchestration<'m> {
    pub fn arrange_waves(&self) -> Vec<Vec<ShaderDependency<'m>>> {
        // rearrange shaders based on their dependencies
        // ideally, you want orchestration to have as few bottlenecks as possible,
        // or as many parallelizable shaders as possible, so you want to arrange the shaders in a way that minimizes the number of shaders that depend on each other

        // we can achieve this with a wave-like algorithm, where we start with shaders that have no dependencies,
        // count the dependencies of each shader, and then iteratively add shaders with the least dependencies until we have arranged all shaders

        let mut arranged: Vec<Vec<ShaderDependency<'m>>> = Vec::new();
        let mut dependencies: HashMap<ShaderDependency<'m>, Vec<ShaderDependency<'m>>> =
            HashMap::new();
        // for shader in &self.shaders {
        //     dependencies.entry(*shader).or_default();
        //     for input in &shader.inputs {
        //         dependencies.entry(*input.shader).or_default().push(*shader);
        //     }
        // }
        let mut agenda: Vec<ShaderDependency<'m>> = self
            .main_shaders
            .iter()
            .cloned()
            .collect::<Vec<ShaderDependency<'m>>>();
        while let Some(dep) = agenda.pop() {
            //dependencies.entry(dep.clone()).or_default();
            if dependencies.contains_key(&dep) {
                continue;
            }
            dependencies.insert(dep.clone(), Vec::new());
            for input in &dep.shader.inputs {
                agenda.push(input.clone());
                dependencies.get_mut(&dep).unwrap().push(input.clone());
            }
        }

        // check if there is a dependency that is not in the dependencies map
        for deps in dependencies.values() {
            for dep in deps {
                if !dependencies.contains_key(dep) {
                    println!("Dependency {:?} is not in the dependencies map", dep.shader);
                }
            }
        }

        while dependencies.len() > 0 {
            let mut wave: Vec<ShaderDependency<'m>> = Vec::new();
            for (shader, deps) in &dependencies {
                if deps.len() == 0 {
                    wave.push(shader.clone());
                }
            }
            if wave.len() == 0 {
                let mut printer = Printer::new();
                //pretty_dependency_map(&dependencies, &mut printer);
                let orchestration_output = printer.finish();
                std::fs::write("orchestration.dot", orchestration_output)
                    .expect("Unable to write file");

                // check if there is a dependency that is not in the dependencies map
                // for deps in dependencies.values() {
                //     for dep in deps {
                //         if !dependencies.contains_key(dep) {
                //             println!("Dependency {:?} is not in the dependencies map", dep.shader);
                //         }
                //     }
                // }

                panic!("Cyclic dependency detected");
            }
            for shader in &wave {
                dependencies.remove(&shader);
                for dependent in dependencies.values_mut() {
                    dependent.retain(|s| s != shader);
                }
            }
            arranged.push(wave);
        }
        //arranged.reverse(); // reverse to get correct order (shaders with no dependencies should be first)
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
                let node_id = p.anon_name(shader.shader, "shader");

                if emitted.insert((shader.shader).addr()) {
                    p.line(&format!(
                        r#"{id} [label="{label}"];"#,
                        id = node_id,
                        label = shader.shader.name
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
                let to_id = p.anon_name(input.shader, "shader");

                p.line(&format!(r#"{from} -> {to};"#, from = from_id, to = to_id));
            }
        }

        p.dedent();
        p.line("}");
    }

    pub fn get_primary_shaders(&self) -> Vec<ShaderDependency<'m>> {
        // shaders that are not depended on by any other shader
        let mut depended_on = HashSet::new();
        for shader in &self.main_shaders {
            for input in &shader.shader.inputs {
                depended_on.insert(input.clone());
            }
        }
        self.main_shaders
            .iter()
            .filter(|s| !depended_on.contains(s))
            .cloned()
            .collect()
    }
}
