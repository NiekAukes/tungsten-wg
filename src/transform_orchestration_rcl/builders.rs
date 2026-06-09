use std::collections::HashMap;
use std::rc::Rc;

use crate::{
    orchestrate::{
        Flatten,
        model::{ShaderDependency, ShaderRef},
    },
    rcl::{Expression, Statement, Struct, Type, Variable},
    spmt::model::PermutationTableInput,
    transform_rcl::{
        BASE3D_NOISE_SAMPLER_STRUCT_NAME, PERLIN_NOISE_SAMPLER_STRUCT_NAME,
        PERM_TABLES_STRUCT_NAME, sanitize_name,
    },
};

pub const PERM_TABLES_PARAM_NAME: &str = "perm_tables";

/// RCL type used for a single boxed permutation table (`Box<[i8; 256]>`).
pub fn perm_table_type(pt: &PermutationTableInput) -> Type {
    match pt {
        PermutationTableInput::PerlinNoise { .. } => {
            Type::Struct(format!("Box<{}>", PERLIN_NOISE_SAMPLER_STRUCT_NAME))
        }
        PermutationTableInput::Base3DNoise => {
            Type::Struct(format!("Box<{}>", BASE3D_NOISE_SAMPLER_STRUCT_NAME))
        }
    }
}

/// Builds an RCL function `make_permutation_tables() -> PermutationTables` that
/// constructs every permutation table by calling `make_permutation_table(i64,i64,i64,i64)`
/// with seeds derived from the MD5 hashes of each table's ident and subident strings.
pub fn build_perm_tables_init_fn<'m>(
    perm_tables: &[PermutationTableInput],
) -> crate::rcl::model::Function<'m> {
    let mut f = crate::rcl::model::Function::new(
        Some("make_permutation_tables".to_string()),
        Type::Struct(PERM_TABLES_STRUCT_NAME.to_string()),
    );

    f.add_parameter("seed".to_string(), Type::I64);
    let seed_var = Rc::new(Variable {
        name: Some("seed".to_string()),
        t: Type::I64,
        mutable: false,
    });

    let mut struct_fields = Vec::new();
    for perm_table in perm_tables {
        match perm_table {
            PermutationTableInput::PerlinNoise {
                ident,
                subident,
                subident_index,
            } => {
                let field_name = perm_table_var_name(perm_table);
                let ident_seed = super::random::xoroshiro_seed(ident);
                let subident_seed = subident
                    .as_deref()
                    .map(super::random::xoroshiro_seed)
                    .unwrap_or((0, 0));

                let call = Expression::LateBoundCall {
                    function_name: "make_permutation_table".to_string(),
                    arguments: vec![
                        Expression::Variable(seed_var.clone()),
                        Expression::I64Literal(ident_seed.0 as i64),
                        Expression::I64Literal(ident_seed.1 as i64),
                        Expression::I64Literal(*subident_index as i64),
                        Expression::I64Literal(subident_seed.0 as i64),
                        Expression::I64Literal(subident_seed.1 as i64),
                    ],
                    argument_types: vec![Type::I64, Type::I64, Type::I64, Type::I64],
                    return_type: perm_table_type(perm_table),
                };
                struct_fields.push((field_name, call));
            }
            PermutationTableInput::Base3DNoise => {
                let field_name = perm_table_var_name(perm_table);
                let call = Expression::LateBoundCall {
                    function_name: "make_base3d_perm_table".to_string(),
                    arguments: vec![Expression::Variable(seed_var.clone())],
                    argument_types: vec![Type::I64],
                    return_type: perm_table_type(perm_table),
                };
                struct_fields.push((field_name, call));
            }
        }
    }

    f.body.push(crate::rcl::model::Statement::Return(Some(
        Expression::StructInit {
            struct_name: PERM_TABLES_STRUCT_NAME.to_string(),
            fields: struct_fields,
        },
    )));

    f
}

/// Builds the `PermutationTables` struct definition from a deduplicated list
/// of permutation tables.
pub fn build_perm_tables_struct(perm_tables: &[PermutationTableInput]) -> Struct {
    let mut s = Struct::new(format!("{}", PERM_TABLES_STRUCT_NAME));
    for perm_table in perm_tables {
        s.add_field(perm_table_var_name(perm_table), perm_table_type(perm_table));
    }
    s
}

/// Derives the Rust field/variable name for a permutation table.
pub fn perm_table_var_name(perm_table: &PermutationTableInput) -> String {
    match perm_table {
        PermutationTableInput::PerlinNoise {
            ident,
            subident,
            subident_index,
        } => sanitize_name(&format!(
            "{}_{}_{}",
            ident,
            subident_index,
            subident.as_ref().unwrap_or(&"".to_string())
        )),
        PermutationTableInput::Base3DNoise => "base3d_perm_table".to_string(),
    }
}

/// Creates a `Box<[f64; N]>` initialiser expression via `make_buffer::<N>()`.
pub fn box_array<'m>(count: usize) -> Expression<'m> {
    Expression::LateBoundCall {
        function_name: format!("make_buffer::<{}>", count),
        arguments: vec![],
        argument_types: vec![],
        return_type: Type::Struct(format!("Box<[f64; {}]>", count)),
    }
}

/// Allocates the output buffer variable for a shader and returns both the
/// `Rc<Variable>` and its corresponding `Declare` statement.
pub fn make_output_buffer<'m>(
    shader_name: &str,
    dimensions: (i32, i32, i32),
) -> (Rc<Variable>, Statement<'m>) {
    let dims = dimensions.flatten() as usize;
    let output_var = Rc::new(Variable {
        name: Some(format!("{}_{}_output", shader_name, dims)),
        t: Type::Struct(format!("Box<[f64; {}]>", dims)),
        mutable: true,
    });

    let declare = Statement::Declare {
        variable: output_var.clone(),
        init: Some(box_array(dims)),
        mutable: true,
    };
    (output_var, declare)
}

/// Builds the `&output` reference arguments and their types for all upstream
/// shader dependencies that feed into the current shader.
pub fn collect_dep_args<'m>(
    shader_inputs: &[ShaderDependency<'m>],
    shader_output_map: &HashMap<ShaderDependency<'m>, Rc<Variable>>,
) -> (Vec<Expression<'m>>, Vec<Type>) {
    let mut dep_exprs = Vec::new();
    let mut dep_types = Vec::new();
    for dep in shader_inputs {
        let dep_output_var = shader_output_map.get(&dep).unwrap();
        dep_exprs.push(Expression::Ref(Box::new(Expression::Variable(
            dep_output_var.clone(),
        ))));
        let dep_dims = dep.dimensions.flatten() as usize;
        dep_types.push(Type::Array(Box::new(Type::F64), dep_dims));
    }
    (dep_exprs, dep_types)
}

/// Produces field-access expressions (`perm_tables.field_name`) for each
/// permutation table a shader needs, and collects the raw descriptors for
/// later deduplication into the `PermutationTables` struct.
pub fn collect_perm_args<'m>(
    perm_tables: &[PermutationTableInput],
    perm_tables_var: Rc<Variable>,
) -> (Vec<Expression<'m>>, Vec<PermutationTableInput>) {
    let mut perm_exprs = Vec::new();
    let mut collected_tables = Vec::new();
    for perm_table in perm_tables {
        let field_name = perm_table_var_name(perm_table);
        perm_exprs.push(Expression::Ref(Box::new(Expression::Field {
            base: Box::new(Expression::Variable(perm_tables_var.clone())),
            field: field_name,
        })));
        collected_tables.push(perm_table.clone());
    }
    (perm_exprs, collected_tables)
}

/// Constructs the `for p in iter_3d(...) { output[as_index(p)] = shader(...) }`
/// loop statement for a single shader dispatch.
pub fn make_shader_loop<'m>(
    dep: &ShaderDependency<'m>,
    output_var: Rc<Variable>,
    origin_var: Rc<Variable>,
    dep_exprs: Vec<Expression<'m>>,
    dep_types: Vec<Type>,
    perm_exprs: Vec<Expression<'m>>,
    wave_i: usize,
    shader_j: usize,
) -> Statement<'m> {
    let dims = dep.dimensions.flatten() as usize;
    let shader_name = sanitize_name(&dep.shader.name);

    let loop_var = Rc::new(Variable {
        name: Some(format!("p_{}_{}", wave_i, shader_j)),
        t: Type::Struct("Pos3".to_string()),
        mutable: false,
    });

    let loop_iter = Expression::LateBoundCall {
        function_name: "iter_3d".to_string(),
        arguments: vec![
            Expression::I32Literal(dep.dimensions.0),
            Expression::I32Literal(dep.dimensions.1),
            Expression::I32Literal(dep.dimensions.2),
        ],
        argument_types: vec![Type::I32, Type::I32, Type::I32],
        return_type: Type::Struct("Pos3".to_string()),
    };

    // Extract scale values from ShaderDependency
    let (origin_scale_x, origin_scale_y, origin_scale_z) = dep.scaled_origin.as_float();
    let (position_scale_x, position_scale_y, position_scale_z) = dep.scaled_position.as_float();

    let origin_scale_expr = Expression::LateBoundCall {
        function_name: "Vec3::new".to_string(),
        arguments: vec![
            Expression::F64Literal(origin_scale_x as f64),
            Expression::F64Literal(origin_scale_y as f64),
            Expression::F64Literal(origin_scale_z as f64),
        ],
        argument_types: vec![Type::F64, Type::F64, Type::F64],
        return_type: Type::Struct("Vec3".to_string()),
    };

    let position_scale_expr = Expression::LateBoundCall {
        function_name: "Vec3::new".to_string(),
        arguments: vec![
            Expression::F64Literal(position_scale_x as f64),
            Expression::F64Literal(position_scale_y as f64),
            Expression::F64Literal(position_scale_z as f64),
        ],
        argument_types: vec![Type::F64, Type::F64, Type::F64],
        return_type: Type::Struct("Vec3".to_string()),
    };

    let mut call_args = vec![
        Expression::Variable(loop_var.clone()),
        Expression::Variable(origin_var),
        origin_scale_expr,
        position_scale_expr,
    ];
    call_args.extend(dep_exprs);
    call_args.extend(perm_exprs.clone());

    let mut call_arg_types = vec![
        Type::Struct("Pos3".to_string()),
        Type::Struct("Vec3".to_string()),
        Type::Struct("Vec3".to_string()),
        Type::Struct("Vec3".to_string()),
    ];
    call_arg_types.extend(dep_types);
    for _ in &perm_exprs {
        //wrong, but we don't actually need the exact types here
        call_arg_types.push(perm_table_type(&PermutationTableInput::Base3DNoise));
    }

    let call_stmt = Statement::ArrayAssign {
        target: output_var,
        index: Expression::LateBoundCall {
            function_name: "as_index".to_string(),
            arguments: vec![
                Expression::Variable(loop_var.clone()),
                Expression::I32Literal(dep.dimensions.2),
                Expression::I32Literal(dep.dimensions.1),
            ],
            argument_types: vec![Type::Struct("Pos3".to_string()), Type::I32, Type::I32],
            return_type: Type::U64,
        },
        value: Expression::LateBoundCall {
            function_name: shader_name,
            arguments: call_args,
            argument_types: call_arg_types,
            return_type: Type::Array(Box::new(Type::F64), dims),
        },
    };

    Statement::ForIn {
        variable: loop_var,
        iterable: loop_iter,
        body: vec![call_stmt],
    }
}
