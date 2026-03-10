use std::{collections::HashMap, rc::Rc};

use crate::{
    orchestrate::{self, Flatten, model::ShaderDependency},
    rcl::{Expression, RCL, Statement, Type},
    spmt::model::Interned,
    transform_rcl::sanitize_name,
};

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
        // convert the orchestration to RCL
        // for each shader, create a function in RCL
        // for each dependency, create a call to the dependent shader's function
        // handle the scaled origin and dimensions as needed

        // create a new function for the orchestration
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

        // add an origin parameter to the orchestration function
        orch_function.add_parameter("origin".to_string(), Type::Struct("Vec3".to_string()));
        let origin_var = Rc::new(crate::rcl::model::Variable {
            name: Some("origin".to_string()),
            t: Type::Struct("Vec3".to_string()),
            mutable: false,
        });

        let mut shader_output_map = HashMap::new();
        for (i, shader_deps) in orchestration.iter().enumerate() {
            // for each wave, get each shader and create an output variable for it
            for (j, dep) in shader_deps.iter().enumerate() {
                let shader_name = sanitize_name(&dep.shader.name);
                let dims = dep.dimensions.flatten();
                let output_var = Rc::new(crate::rcl::model::Variable {
                    name: Some(format!("{}_output", shader_name)),
                    t: Type::Struct(format!("Box<[f32; {}]>", dims)),
                    mutable: true,
                });
                orch_function.add_statement(Statement::Declare {
                    variable: output_var.clone(),
                    // init: Some(Expression::ArrayInit {
                    //     element_type: Type::F32,
                    //     element: Box::new(Expression::FloatLiteral(0.0)),
                    //     count: dims as usize,
                    // }),
                    init: Some(self.box_array(Expression::FloatLiteral(0.0), dims as usize)),
                    mutable: true,
                });
                shader_output_map.insert(dep.shader, output_var.clone());

                // get the dependencies of this shader and create calls to them
                let mut dependencies = Vec::new();
                let mut dep_types = Vec::new();
                for dep in &dep.shader.inputs {
                    let dep_output_var = shader_output_map.get(&dep.shader).unwrap();
                    //dependencies.push(dep_output_var.clone());
                    dependencies.push(Expression::Ref(Box::new(Expression::Variable(
                        dep_output_var.clone(),
                    ))));

                    // also calculate the type of the dependency output variable
                    let dep_dims = dep.dimensions.flatten();
                    dep_types.push(Type::Array(Box::new(Type::F32), dep_dims as usize));
                }

                // create a for loop to iterate over the dimensions and call the shader function for each element
                let loop_var_name = format!("p_{}_{}", i, j);
                let loop_var = Rc::new(crate::rcl::model::Variable {
                    name: Some(loop_var_name.clone()),
                    t: Type::Struct("Pos3".to_string()),
                    mutable: false,
                });

                let loop_iter = Expression::LateBoundCall {
                    function_name: "iter_3d".to_string(),
                    arguments: vec![
                        Expression::IntLiteral(dep.dimensions.0 as i64),
                        Expression::IntLiteral(dep.dimensions.1 as i64),
                        Expression::IntLiteral(dep.dimensions.2 as i64),
                    ],
                    argument_types: vec![Type::I32, Type::I32, Type::I32],
                    return_type: Type::Struct("Pos3".to_string()),
                };

                let mut fdeps = Vec::new();
                fdeps.push(Expression::Variable(loop_var.clone()));
                fdeps.push(Expression::Variable(origin_var.clone()));
                fdeps.extend(dependencies.clone());
                let mut fdep_types = vec![
                    Type::Struct("Pos3".to_string()),
                    Type::Struct("Vec3".to_string()),
                ];
                fdep_types.extend(dep_types.clone());

                let call_stmt = crate::rcl::model::Statement::ArrayAssign {
                    target: output_var.clone(),
                    index: Expression::LateBoundCall {
                        function_name: "as_index".to_string(),
                        arguments: vec![
                            Expression::Variable(loop_var.clone()),
                            Expression::IntLiteral(dep.dimensions.2 as i64),
                            Expression::IntLiteral(dep.dimensions.1 as i64),
                        ],
                        argument_types: vec![
                            Type::Struct("Pos3".to_string()),
                            Type::I32,
                            Type::I32,
                        ],
                        return_type: Type::U64,
                    },
                    value: Expression::LateBoundCall {
                        function_name: sanitize_name(&dep.shader.name),
                        arguments: fdeps,
                        argument_types: fdep_types, // TODO: handle argument types based on dependencies
                        return_type: Type::Array(Box::new(Type::F32), dims as usize),
                    },
                };

                let lp = Statement::ForIn {
                    variable: loop_var.clone(),
                    iterable: loop_iter,
                    body: vec![call_stmt],
                };
                orch_function.body.push(lp);
            }
        }

        // handle the return values of the orchestration function
        let mut return_types = Vec::new();
        let mut return_values = Vec::new();
        for dep in returns {
            let output_var = shader_output_map.get(&dep.shader).unwrap();
            return_types.push(output_var.t.clone());
            return_values.push(Expression::Variable(output_var.clone()));
        }

        // return a tuple of the return values
        orch_function.return_type = Type::Tuple(return_types);
        orch_function
            .body
            .push(Statement::Return(Some(Expression::TupleInit(
                return_values,
            ))));

        self.rcl_model
            .main_functions
            .push(Interned::new(self.arena.alloc(orch_function)));

        self.rcl_model
    }

    fn box_array(&self, elem: Expression<'m>, count: usize) -> Expression<'m> {
        Expression::LateBoundCall {
            function_name: format!("make_buffer::<{}>", count),
            // arguments: vec![Expression::ArrayInit {
            //     element_type: Type::F32,
            //     element: Box::new(elem),
            //     count,
            // }],
            arguments: vec![],
            argument_types: vec![],
            //argument_types: vec![Type::Array(Box::new(Type::F32), count)],
            return_type: Type::Struct(format!("Box<[f32; {}]>", count)),
        }
    }

    fn make_vec_repeat(&self, elem: Expression<'m>, count: usize) -> Expression<'m> {
        let v = match elem {
            Expression::FloatLiteral(f) => format!("vec![{}f32; {}]", f, count),
            Expression::IntLiteral(i) => format!("vec![{}i32; {}]", i, count),
            _ => panic!("Unsupported element type for make_vec_repeat"),
        };
        Expression::InlineRust(v)
    }

    fn make_vec(elems: Vec<Expression<'m>>, t: Type) -> Expression<'m> {
        let types = vec![t.clone(); elems.len()];
        Expression::LateBoundCall {
            function_name: "Vec::new".to_string(),
            arguments: elems,
            argument_types: types,
            return_type: Type::Struct("Vec".to_string()),
        }
    }
}
