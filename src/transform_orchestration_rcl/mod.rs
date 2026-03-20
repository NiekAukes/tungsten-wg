use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    orchestrate::model::ShaderDependency,
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
}

impl<'m> OrchestrationConverter<'m> {
    pub fn new(arena: &'m bumpalo::Bump) -> Self {
        OrchestrationConverter {
            rcl_model: RCL::new(),
            arena,
        }
    }

    pub fn convert(
        mut self,
        orchestration: Vec<Vec<ShaderDependency<'m>>>,
        returns: Vec<ShaderDependency<'m>>,
    ) -> RCL<'m> {
        let mut orch_function = crate::rcl::model::Function {
            name: Some("orchestration".to_string()),
            parameters: Vec::new(),
            variables: Vec::new(),
            body: Vec::new(),
            return_type: crate::rcl::model::Type::Void,
            inline: false,
        };

        self.rcl_model
            .add_import("crate::density_function::*".to_string());

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

        let mut all_perm_tables = Vec::new();
        let mut shader_output_map = HashMap::new();

        for (i, shader_deps) in orchestration.iter().enumerate() {
            for (j, dep) in shader_deps.iter().enumerate() {
                let shader_name = sanitize_name(&dep.shader.name);

                let (output_var, declare) =
                    builders::make_output_buffer(&shader_name, dep.dimensions);
                orch_function.add_statement(declare);
                shader_output_map.insert(dep.shader, output_var.clone());

                let (dep_exprs, dep_types) =
                    builders::collect_dep_args(&dep.shader.inputs, &shader_output_map);

                let (perm_exprs, perm_tables) = builders::collect_perm_args(
                    &dep.shader.permutation_tables,
                    perm_tables_var.clone(),
                );
                all_perm_tables.extend(perm_tables);

                let loop_stmt = builders::make_shader_loop(
                    dep,
                    output_var,
                    origin_var.clone(),
                    dep_exprs,
                    dep_types,
                    perm_exprs,
                    i,
                    j,
                );
                orch_function.body.push(loop_stmt);
            }
        }

        let mut seen = HashSet::new();
        let deduped_perm_tables: Vec<_> = all_perm_tables
            .into_iter()
            .filter(|t| seen.insert(builders::perm_table_var_name(t)))
            .collect();

        let perm_tables_struct = builders::build_perm_tables_struct(&deduped_perm_tables);
        self.rcl_model
            .structs
            .push(Interned::new(self.arena.alloc(perm_tables_struct)));

        let perm_tables_init_fn = builders::build_perm_tables_init_fn(&deduped_perm_tables);
        self.rcl_model
            .main_functions
            .push(Interned::new(self.arena.alloc(perm_tables_init_fn)));

        orch_function.add_parameter(
            builders::PERM_TABLES_PARAM_NAME.to_string(),
            Type::Struct(format!("&{}", PERM_TABLES_STRUCT_NAME)),
        );

        let (output_struct, struct_fields) =
            output::build_return_struct(&returns, &shader_output_map);
        self.rcl_model
            .structs
            .push(Interned::new(self.arena.alloc(output_struct)));

        orch_function.return_type = Type::Struct(output::OUTPUT_STRUCT_NAME.to_string());
        orch_function
            .body
            .push(Statement::Return(Some(Expression::StructInit {
                struct_name: output::OUTPUT_STRUCT_NAME.to_string(),
                fields: struct_fields,
            })));

        self.rcl_model
            .main_functions
            .push(Interned::new(self.arena.alloc(orch_function)));

        self.rcl_model
    }
}
