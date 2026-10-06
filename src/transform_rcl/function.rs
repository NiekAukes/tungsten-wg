/*
Function conversion from SPMT to RCL.
Handles conversion of function definitions and density function definitions.
*/

use std::collections::HashMap;
use std::rc::{self, Rc};

use super::{RCLFunctionConverter, sanitize_name, statement, types};
use crate::orchestrate::{Flatten, Scale};
use crate::rcl::{Parameter, Type, model as rcl};
use crate::spmt::model::{self as spmt, Addr, Interned, Name};
use crate::transform_rcl::{HOST_CONSTS_STRUCT_NAME, InputKey};
use crate::transform_rcl::types::{convert_type, permutation_table_var_name};

/// Convert an SPMT helper function to an RCL function.
///
/// The function name is prefixed with the parent density function so helpers are unique per density.
/// Density inputs and `host_consts` are appended as extra parameters.
/// Returns `(function, converter, is_new)`; `is_new` is false when a cached function was reused.
pub fn convert_function<'a, 'm>(
    spmt_func: &spmt::Function<'a>,
    arena: &'m bumpalo::Bump,
    already_converted_functions: HashMap<*const (), rcl::FunctionRef<'m>>,
    density_inputs: Rc<HashMap<InputKey, Parameter>>,
    parent_density_name: String,
    host_fields: Rc<HashMap<InputKey, String>>,
) -> (rcl::FunctionRef<'m>, RCLFunctionConverter<'m>, bool) {
    let mut converter = RCLFunctionConverter::new_with_density_inputs(
        arena,
        density_inputs.clone(),
        parent_density_name.clone(),
    );
    converter.already_converted_functions = already_converted_functions;
    converter.host_fields = host_fields.clone();
    let func_name = spmt_func
        .canonical_name
        .as_deref()
        .map(sanitize_name)
        .unwrap_or_else(|| format!("function_{}", spmt_func.addr() as usize));
    let func_name = format!("{}_{}", parent_density_name, func_name);
    let mut rcl_func =
        rcl::Function::new(Some(func_name), Some(convert_type(&spmt_func.return_type)));

    if let Some(cached) = converter.already_converted_functions.get(&spmt_func.addr()) {
        return (cached.clone(), converter, false);
    }

    // the cache is not enough, need to check if we have a function with the same name
    for (_, func_ref) in &converter.already_converted_functions {
        if func_ref.name.as_deref() == Some(&rcl_func.name.clone().unwrap_or_default()) {
            return (func_ref.clone(), converter, false);
        }
    }

    // Convert parameters
    for (i, param) in spmt_func.parameters.iter().enumerate() {
        let param_type = types::convert_type(&param.t);
        // when param name is empty, generate a unique name based on the parameter index
        let param_name = match &param.name {
            Name::Named(n) => sanitize_name(n),
            Name::Prefixed(prefix) => sanitize_name(&format!("{}_{}", prefix, i)),
            Name::Anonymous => format!("param_{}", i),
        };
        rcl_func.add_parameter(param_name, param_type);
    }

    for (_, input) in density_inputs.as_ref() {
        rcl_func.add_parameter(input.name.clone(), input.t.clone());
    }
    rcl_func.add_parameter(
        "host_consts".to_string(),
        rcl::Type::Struct(format!("&{}", HOST_CONSTS_STRUCT_NAME)),
    );

    // Register variables in converter
    for var in &spmt_func.variables {
        let rcl_var = Rc::new(rcl::Variable {
            name: Some(converter.get_concrete_name(var.clone())),
            t: types::convert_type(&var.t),
            mutable: true,
        });
        converter.register_variable(var.clone(), rcl_var.clone());
        rcl_func.add_variable(rcl_var);
    }

    // Convert body statements
    for stmt in &spmt_func.body {
        let converted_stmt = converter.convert_statement(stmt);
        rcl_func.add_statement(converted_stmt);
    }

    let rcl_func_ref = Interned::new(arena.alloc(rcl_func));
    // Store the converted function in the map
    converter
        .already_converted_functions
        .insert(spmt_func.addr(), rcl_func_ref.clone());

    (rcl_func_ref, converter, true)
}

/// Convert an SPMT density function to an RCL function.
///
/// The generated function loops over every position in `dimensions`, evaluates the body,
/// and stores the returned value into the flattened `out` array.
/// Returns `(helper functions, main function, converter, constants)`.
pub fn convert_density_function<'a, 'm>(
    spmt_df: spmt::DensityFunctionRef<'a>,
    arena: &'m bumpalo::Bump,
    dimensions: (i32, i32, i32),
    already_converted_functions: HashMap<*const (), rcl::FunctionRef<'m>>,
    host_fields: Rc<HashMap<InputKey, String>>,
) -> (
    Vec<rcl::FunctionRef<'m>>,
    rcl::FunctionRef<'m>,
    RCLFunctionConverter<'m>,
    Vec<rcl::Constant<'m>>,
) {
    let mut rcl_funcs = Vec::new();
    // let density_func_name = spmt_df
    //     .canonical_name
    //     .as_deref()
    //     .map(sanitize_name)
    //     .unwrap_or_else(|| format!("density_function_{}", spmt_df.addr() as usize));
    let density_func_name = derive_density_function_name(spmt_df, dimensions);
    let mut rcl_func = rcl::Function::new(Some(density_func_name.clone()), None);

    // Output buffer, plus origin/scale parameters.
    //rcl_func.add_parameter("pos3".to_string(), rcl::Type::Struct("Pos3".to_string()));
    rcl_func.add_parameter(
        "out".to_string(),
        rcl::Type::MutArrayRef(Box::new(rcl::Type::F64), dimensions.flatten() as usize),
    );
    let out_var = Rc::new(rcl::Variable {
        name: Some("out".to_string()),
        t: rcl::Type::MutArrayRef(Box::new(rcl::Type::F64), dimensions.flatten() as usize),
        mutable: true,
    });

    let p_var = Rc::new(rcl::Variable {
        name: Some("p".to_string()),
        t: rcl::Type::Struct("PositionIterator".to_string()),
        mutable: false,
    });

    rcl_func.add_statement(rcl::Statement::InlineRust("let pos3 = p.pos;".to_string()));

    rcl_func.add_parameter("origin".to_string(), rcl::Type::Struct("Vec3".to_string()));
    rcl_func.add_parameter(
        "origin_scale".to_string(),
        rcl::Type::Struct("Vec3".to_string()),
    );
    rcl_func.add_parameter(
        "position_scale".to_string(),
        rcl::Type::Struct("Vec3".to_string()),
    );
    // Add density inputs as parameters
    let mut density_inputs = HashMap::new();
    for (i, input) in spmt_df.density_inputs.iter().enumerate() {
        // let param_name = sanitize_name(
        //     &input
        //         .var
        //         .name
        //         .clone()
        //         .unwrap_or_else(|| format!("input_{}", input.density_function.addr() as usize)),
        // );
        let param_name = match &input.var.name {
            Name::Named(n) => sanitize_name(n),
            Name::Prefixed(prefix) => sanitize_name(&format!(
                "{}_{}",
                prefix,
                input.density_function.addr() as usize
            )),
            Name::Anonymous => format!("input_{}", input.density_function.addr() as usize),
        };

        // Single-cell inputs are passed by value, larger ones as an array reference.
        let dimensions = input.dimensions.flatten();
        if dimensions == 1 {
            rcl_func.add_parameter(
                param_name.clone(),
                convert_type(&spmt::VariableType::DensityInput),
            );
            density_inputs.insert(
                InputKey::from(input),
                Parameter {
                    name: param_name,
                    t: convert_type(&spmt::VariableType::DensityInput),
                },
            );
        } else {
            rcl_func.add_parameter(
                param_name.clone(),
                rcl::Type::ArrayRef(
                    Box::new(convert_type(&spmt::VariableType::DensityInput)),
                    dimensions as usize,
                ),
            );
            density_inputs.insert(
                InputKey::from(input),
                Parameter {
                    name: param_name,
                    t: rcl::Type::ArrayRef(
                        Box::new(convert_type(&spmt::VariableType::DensityInput)),
                        input.dimensions.flatten() as usize,
                    ),
                },
            );
        };
    }

    // add permutation tables as parameters
    //let mut perm_tables = Vec::new();
    for input in &spmt_df.permutation_table_inputs {
        let param_name = permutation_table_var_name(input);
        rcl_func.add_parameter(param_name.clone(), get_permutation_table_type(input));
        //perm_tables.push(input.clone());
    }

    // add host input parameters
    rcl_func.add_parameter(
        "host_consts".to_string(),
        rcl::Type::Struct(format!("&{}", HOST_CONSTS_STRUCT_NAME)),
    );
    let density_inputs = Rc::new(density_inputs);

    let mut converter = RCLFunctionConverter::new_with_density_inputs(
        arena,
        density_inputs.clone(),
        density_func_name.clone(),
    );
    converter.already_converted_functions = already_converted_functions;
    converter.host_fields = host_fields.clone();

    for f in spmt_df.helper_functions.iter() {
        let (rcl_func, fconv, is_new) = convert_function(
            f,
            arena,
            converter.already_converted_functions,
            density_inputs.clone(),
            density_func_name.clone(),
            host_fields.clone(),
        );
        if is_new {
            rcl_funcs.push(rcl_func);
        }
        converter.already_converted_functions = fconv.already_converted_functions;
    }

    // Register variables in converter
    for var in &spmt_df.variables {
        let rcl_var = Rc::new(rcl::Variable {
            name: Some(converter.get_concrete_name(var.clone())),
            t: types::convert_type(&var.t),
            mutable: true,
        });
        converter.register_variable(var.clone(), rcl_var.clone());
        rcl_func.add_variable(rcl_var);
    }

    // Register constants in converter
    let mut constants = Vec::new();
    for (var, value) in &spmt_df.constants {
        // let varname = var.name.as_deref().map(sanitize_name).map(|name| {
        //     let base_name = spmt_df
        //         .canonical_name
        //         .as_ref()
        //         .map(|cn| sanitize_name(cn))
        //         .unwrap_or_else(|| "density_func".to_string());
        //     format!("{}_{}", base_name, name)
        // });
        let varname = match &var.name {
            Name::Named(n) => sanitize_name(n),
            Name::Prefixed(prefix) => {
                let base_name = spmt_df
                    .canonical_name
                    .as_ref()
                    .map(|cn| sanitize_name(cn))
                    .unwrap_or_else(|| "density_func".to_string());
                sanitize_name(&format!("{}_{}_{}", base_name, prefix, var.addr() as usize))
            }
            Name::Anonymous => {
                let base_name = spmt_df
                    .canonical_name
                    .as_ref()
                    .map(|cn| sanitize_name(cn))
                    .unwrap_or_else(|| "density_func".to_string());
                format!("{}_const_{}", base_name, var.addr() as usize)
            }
        };

        let rcl_var = Rc::new(rcl::Variable {
            name: Some(varname),
            t: types::convert_type(&var.t),
            mutable: false,
        });

        converter.register_variable(var.clone(), rcl_var.clone());
        constants.push(rcl::Constant {
            var: rcl_var,
            value: converter.convert_expression(value),
        });
    }

    // Convert body statements
    for stmt in &spmt_df.body {
        let converted_stmt = converter.convert_statement(stmt);
        rcl_func.add_statement(converted_stmt);
    }

    // The final return becomes an array write inside the loop.
    let Some(rcl::Statement::Return(Some(end_expr))) = rcl_func.body.pop() else {
        panic!("Density function must end with a return statement");
    };

    let assign = rcl::Statement::ArrayAssign {
        target: out_var,
        index: pos3_index(p_var.clone()),
        value: end_expr,
    };
    rcl_func.add_statement(assign);

    let end_stmt = wrap_in_loop(rcl_func.body, p_var.clone(), dimensions);
    rcl_func.body = Vec::new();
    rcl_func.add_statement(end_stmt);

    let rcl_func_ref = Interned::new(arena.alloc(rcl_func));
    // Store the converted function in the map
    converter
        .already_converted_functions
        .insert(spmt_df.addr(), rcl_func_ref.clone());

    (rcl_funcs, rcl_func_ref, converter, constants)
}

/// Parameter type for a permutation table: a reference to the matching sampler struct.
fn get_permutation_table_type(input: &spmt::PermutationTableInput) -> rcl::Type {
    match input {
        spmt::PermutationTableInput::PerlinNoise { .. } => rcl::Type::Ref(Box::new(
            rcl::Type::Struct(crate::transform_rcl::PERLIN_NOISE_SAMPLER_STRUCT_NAME.to_string()),
        )),
        spmt::PermutationTableInput::Base3DNoise { .. } => rcl::Type::Ref(Box::new(
            rcl::Type::Struct(crate::transform_rcl::BASE3D_NOISE_SAMPLER_STRUCT_NAME.to_string()),
        )),
    }
}

/// Wrap `body` in a `for pos3 in iter_3d(x, y, z)` loop over the given dimensions.
fn wrap_in_loop<'a>(
    body: Vec<rcl::Statement<'a>>,
    pos3_var: Rc<rcl::Variable>,
    dimensions: (i32, i32, i32),
) -> rcl::Statement<'a> {
    // for pos3 in iter_3d(5_i32, 49_i32, 5_i32) {
    // statements...
    // }
    let iter_call = rcl::Expression::LateBoundCall {
        function_name: "iter_3d".to_string(),
        argument_types: vec![rcl::Type::I32, rcl::Type::I32, rcl::Type::I32],
        return_type: rcl::Type::Struct("Pos3".to_string()),
        arguments: vec![
            rcl::Expression::I32Literal(dimensions.0),
            rcl::Expression::I32Literal(dimensions.1),
            rcl::Expression::I32Literal(dimensions.2),
        ],
    };
    rcl::Statement::ForIn {
        variable: pos3_var,
        iterable: iter_call,
        body,
    }
}

/// Flat index of `pos3` into the output array: `x * dim_y * dim_z + y * dim_z + z`.
fn pos3_index<'m>(pos3_var: Rc<rcl::Variable>) -> rcl::Expression<'m> {
    rcl::Expression::LateBoundCall {
        function_name: "index".to_string(),
        argument_types: vec![rcl::Type::Struct("Pos3".to_string())],
        return_type: rcl::Type::I32,
        arguments: vec![rcl::Expression::Variable(pos3_var)],
    }
}

/// Name of the generated function, e.g. `foo_d16x16x16`; dimensions keep per-size variants distinct.
fn derive_density_function_name(spmt_df: spmt::DensityFunctionRef<'_>, dimensions: (i32, i32, i32)) -> String {
        let density_func_name = spmt_df
        .canonical_name
        .as_deref()
        .map(sanitize_name)
        .unwrap_or_else(|| format!("density_function_{}", spmt_df.addr() as usize));
    format!("{}_d{}x{}x{}", density_func_name, dimensions.0, dimensions.1, dimensions.2)
}