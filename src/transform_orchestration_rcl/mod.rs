use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    orchestrate::{Flatten, model::ShaderDependency},
    rcl::{Expression, RCL, Statement, Type, Variable},
    spmt::model::Interned,
    transform_rcl::{self, PERM_TABLES_STRUCT_NAME, sanitize_name},
};

mod builders;
mod output;
mod random;

pub struct OrchestrationConverter<'m> {
    rcl_model: RCL<'m>,
    arena: &'m bumpalo::Bump,
    /// All collected perm tables across all generated functions (for dedup).
    all_perm_tables: Vec<crate::spmt::model::PermutationTableInput>,
}

impl<'m> OrchestrationConverter<'m> {
    pub fn new(arena: &'m bumpalo::Bump) -> Self {
        OrchestrationConverter {
            rcl_model: RCL::new(),
            arena,
            all_perm_tables: Vec::new(),
        }
    }

    /// Generate the full `orchestration()` function that computes all densities.
    /// Call this first, then optionally call `convert_single_entry()` for each
    /// primary density, then call `finish()` to get the final RCL model.
    pub fn convert(
        &mut self,
        orchestration: &Vec<Vec<ShaderDependency<'m>>>,
        returns: Vec<ShaderDependency<'m>>,
    ) {
        let mut orch_function = crate::rcl::model::Function {
            name: Some("orchestration".to_string()),
            parameters: Vec::new(),
            variables: Vec::new(),
            body: Vec::new(),
            return_type: Some(crate::rcl::model::Type::Void),
            inline: false,
        };

        self.rcl_model
            .add_import("super::density_function::*".to_string());

        orch_function.add_parameter("origin".to_string(), Type::Struct("Vec3".to_string()));
        let origin_var = Rc::new(Variable {
            name: Some("origin".to_string()),
            t: Type::Struct("Vec3".to_string()),
            mutable: false,
        });

        let perm_tables_var = Rc::new(Variable {
            name: Some(builders::PERM_TABLES_PARAM_NAME.to_string()),
            t: Type::Struct(format!("&{}", PERM_TABLES_STRUCT_NAME)),
            mutable: false,
        });

        let mut shader_output_map = HashMap::new();

        for (i, shader_deps) in orchestration.iter().enumerate() {
            for (j, dep) in shader_deps.iter().enumerate() {
                if dep.dimensions.flatten() == 1 {
                    self.make_singlesampler_dependency(
                        &mut orch_function,
                        dep,
                        origin_var.clone(),
                        perm_tables_var.clone(),
                        i,
                        j,
                        &mut shader_output_map,
                    );
                } else {
                    self.make_multisampler_dependency(
                        &mut orch_function,
                        dep,
                        origin_var.clone(),
                        perm_tables_var.clone(),
                        i,
                        j,
                        &mut shader_output_map,
                    );
                }
            }
        }

        orch_function.add_parameter(
            builders::PERM_TABLES_PARAM_NAME.to_string(),
            Type::Struct(format!("&{}", PERM_TABLES_STRUCT_NAME)),
        );

        let (output_struct, struct_fields) =
            output::build_return_struct(&returns, &shader_output_map);
        self.rcl_model
            .structs
            .push(Interned::new(self.arena.alloc(output_struct)));

        orch_function.return_type = Some(Type::Struct(output::OUTPUT_STRUCT_NAME.to_string()));
        orch_function
            .body
            .push(Statement::Return(Some(Expression::StructInit {
                struct_name: output::OUTPUT_STRUCT_NAME.to_string(),
                fields: struct_fields,
            })));

        self.rcl_model
            .main_functions
            .push(Interned::new(self.arena.alloc(orch_function)));
    }

    /// Generate a pruned orchestration function for a single entry density.
    /// The function is named `orchestrate_{name}` and returns a `Box<[f64; N]>`
    /// containing only the output of the target density and its transitive deps.
    pub fn convert_single_entry(
        &mut self,
        name: &str,
        waves: Vec<Vec<ShaderDependency<'m>>>,
        target: &ShaderDependency<'m>,
    ) {
        let fn_name = format!("orchestrate_{}", sanitize_name(name));
        let dims = target.dimensions.flatten() as usize;
        let single_sampler_mode = dims == 1;
        let return_type = if single_sampler_mode {
            Type::F64
        } else {
            Type::Struct(format!("Box<[f64; {}]>", dims))
        };

        let mut func = crate::rcl::model::Function {
            name: Some(fn_name),
            parameters: Vec::new(),
            variables: Vec::new(),
            body: Vec::new(),
            return_type: Some(return_type.clone()),
            inline: false,
        };

        func.add_parameter("origin".to_string(), Type::Struct("Vec3".to_string()));
        let origin_var = Rc::new(Variable {
            name: Some("origin".to_string()),
            t: Type::Struct("Vec3".to_string()),
            mutable: false,
        });

        let perm_tables_var = Rc::new(Variable {
            name: Some(builders::PERM_TABLES_PARAM_NAME.to_string()),
            t: Type::Struct(format!("&{}", PERM_TABLES_STRUCT_NAME)),
            mutable: false,
        });
        func.add_parameter(
            builders::PERM_TABLES_PARAM_NAME.to_string(),
            Type::Struct(format!("&{}", PERM_TABLES_STRUCT_NAME)),
        );

        let mut shader_output_map = HashMap::new();

        for (i, shader_deps) in waves.iter().enumerate() {
            for (j, dep) in shader_deps.iter().enumerate() {
                if dep.dimensions.flatten() == 1 {
                    self.make_singlesampler_dependency(
                        &mut func,
                        dep,
                        origin_var.clone(),
                        perm_tables_var.clone(),
                        i,
                        j,
                        &mut shader_output_map,
                    );
                } else {
                    self.make_multisampler_dependency(
                        &mut func,
                        dep,
                        origin_var.clone(),
                        perm_tables_var.clone(),
                        i,
                        j,
                        &mut shader_output_map,
                    );
                }
            }
        }

        // Return the target density's output buffer directly
        let target_output = shader_output_map.get(target).unwrap();
        func.body.push(Statement::Return(Some(Expression::Variable(
            target_output.clone(),
        ))));

        self.rcl_model
            .main_functions
            .push(Interned::new(self.arena.alloc(func)));
    }

    fn make_singlesampler_dependency(
        &mut self,
        func: &mut crate::rcl::model::Function<'m>,
        dep: &ShaderDependency<'m>,
        origin_var: Rc<Variable>,
        perm_tables_var: Rc<Variable>,
        i: usize,
        j: usize,
        shader_output_map: &mut HashMap<ShaderDependency<'m>, Rc<Variable>>,
    ) {
        let shader_name = sanitize_name(&dep.shader.name);

        let output_var = Rc::new(Variable {
            name: Some(format!("{}_output", shader_name)),
            t: Type::F64,
            mutable: false,
        });
        shader_output_map.insert(dep.clone(), output_var.clone());

        let (dep_exprs, dep_types) =
            builders::collect_dep_args(&dep.shader.inputs, &shader_output_map);

        let (perm_exprs, perm_tables) =
            builders::collect_perm_args(&dep.shader.permutation_tables, perm_tables_var.clone());
        self.all_perm_tables.extend(perm_tables);

        let call_expr: Expression<'m> = builders::make_shader_call(
            dep,
            Expression::Construct {
                t: Type::Struct("Pos3".to_string()),
                args: vec![
                    ("x", Expression::I32Literal(0)),
                    ("y", Expression::I32Literal(0)),
                    ("z", Expression::I32Literal(0)),
                ],
            },
            origin_var.clone(),
            dep_exprs,
            dep_types,
            perm_exprs,
            i,
            j,
        );

        let call_stmt = Statement::Declare {
            variable: output_var.clone(),
            init: Some(call_expr),
            mutable: false,
        };
        func.body.push(call_stmt);
    }

    fn make_multisampler_dependency(
        &mut self,
        func: &mut crate::rcl::model::Function<'m>,
        dep: &ShaderDependency<'m>,
        origin_var: Rc<Variable>,
        perm_tables_var: Rc<Variable>,
        i: usize,
        j: usize,
        shader_output_map: &mut HashMap<ShaderDependency<'m>, Rc<Variable>>,
    ) {
        let shader_name = sanitize_name(&dep.shader.name);

        let (output_var, declare) = builders::make_output_buffer(&shader_name, dep.dimensions);
        func.add_statement(declare);
        shader_output_map.insert(dep.clone(), output_var.clone());

        let (dep_exprs, dep_types) =
            builders::collect_dep_args(&dep.shader.inputs, &shader_output_map);

        let (perm_exprs, perm_tables) =
            builders::collect_perm_args(&dep.shader.permutation_tables, perm_tables_var.clone());
        self.all_perm_tables.extend(perm_tables);

        let loop_stmt: Statement<'m> = builders::make_shader_loop(
            dep,
            output_var,
            origin_var.clone(),
            dep_exprs,
            dep_types,
            perm_exprs,
            i,
            j,
        );
        func.body.push(loop_stmt);
    }

    /// Finalise the RCL model: dedup perm tables, build the struct + init fn, return the model.
    pub fn finish(self) -> RCL<'m> {
        let mut seen = HashSet::new();
        let deduped_perm_tables: Vec<_> = self
            .all_perm_tables
            .into_iter()
            .filter(|t| seen.insert(builders::perm_table_var_name(t)))
            .collect();

        let perm_tables_struct = builders::build_perm_tables_struct(&deduped_perm_tables);
        let mut rcl_model = self.rcl_model;
        rcl_model
            .structs
            .push(Interned::new(self.arena.alloc(perm_tables_struct)));

        let perm_tables_init_fn = builders::build_perm_tables_init_fn(&deduped_perm_tables);
        rcl_model
            .main_functions
            .push(Interned::new(self.arena.alloc(perm_tables_init_fn)));

        rcl_model
    }
}
