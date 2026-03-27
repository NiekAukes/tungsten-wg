/*
Function conversion from SPMT to RCL.
Handles conversion of function definitions and density function definitions.
*/

use std::collections::HashMap;
use std::rc::Rc;

use super::{RCLFunctionConverter, sanitize_name, statement, types};
use crate::orchestrate::{Flatten, Scale};
use crate::rcl::{Parameter, Type, model as rcl};
use crate::spmt::model::{self as spmt, Addr, Interned};
use crate::transform_rcl::InputKey;
use crate::transform_rcl::types::{convert_type, permutation_table_var_name};

/// Convert an SPMT function to an RCL function with converter state

pub fn convert_function<'a, 'm>(
    spmt_func: &spmt::Function<'a>,
    arena: &'m bumpalo::Bump,
    already_converted_functions: HashMap<*const (), rcl::FunctionRef<'m>>,
    density_inputs: Rc<HashMap<InputKey, Parameter>>,
) -> (rcl::FunctionRef<'m>, RCLFunctionConverter<'m>) {
    let mut converter =
        RCLFunctionConverter::new_with_density_inputs(arena, density_inputs.clone());
    converter.already_converted_functions = already_converted_functions;
    let mut rcl_func = rcl::Function::new(
        spmt_func.canonical_name.as_deref().map(sanitize_name),
        convert_type(&spmt::VariableType::DensityInput),
    );

    // Convert parameters
    for (i, param) in spmt_func.parameters.iter().enumerate() {
        let param_type = types::convert_type(&param.t);
        // when param name is empty, generate a unique name based on the parameter index
        let param_name =
            sanitize_name(&param.name.clone().unwrap_or_else(|| format!("param{}", i)));
        rcl_func.add_parameter(param_name, param_type);
    }

    for (_, input) in density_inputs.as_ref() {
        rcl_func.add_parameter(input.name.clone(), input.t.clone());
    }

    // Register variables in converter
    for var in &spmt_func.variables {
        let rcl_var = Rc::new(rcl::Variable {
            name: var.name.as_deref().map(sanitize_name),
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

    (rcl_func_ref, converter)
}

/// Convert an SPMT density function to an RCL function with converter state
pub fn convert_density_function<'a, 'm>(
    spmt_df: &'a spmt::DensityFunction<'a>,
    arena: &'m bumpalo::Bump,
    already_converted_functions: HashMap<*const (), rcl::FunctionRef<'m>>,
) -> (
    Vec<rcl::FunctionRef<'m>>,
    rcl::FunctionRef<'m>,
    RCLFunctionConverter<'m>,
) {
    let mut rcl_funcs = Vec::new();
    let mut rcl_func = rcl::Function::new(
        spmt_df.canonical_name.as_deref().map(sanitize_name),
        convert_type(&spmt::VariableType::DensityInput),
    );

    // Add position parameters (x, y, z)
    // and origin parameters (ox, oy, oz)
    rcl_func.add_parameter("pos3".to_string(), rcl::Type::Struct("Pos3".to_string()));
    rcl_func.add_parameter("origin".to_string(), rcl::Type::Struct("Vec3".to_string()));
    // Add density inputs as parameters
    let mut density_inputs = HashMap::new();
    for (i, input) in spmt_df.density_inputs.iter().enumerate() {
        let param_name = sanitize_name(
            &input
                .var
                .name
                .clone()
                .unwrap_or_else(|| format!("input_{}", input.density_function.addr() as usize)),
        );

        let dimensions = input.dimensions.flatten();
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
    }

    // add permutation tables as parameters
    //let mut perm_tables = Vec::new();
    for input in &spmt_df.permutation_table_inputs {
        let param_name = permutation_table_var_name(input);
        rcl_func.add_parameter(
            param_name.clone(),
            convert_type(&spmt::VariableType::PermutationTable),
        );
        //perm_tables.push(input.clone());
    }

    let density_inputs = Rc::new(density_inputs);

    let mut converter =
        RCLFunctionConverter::new_with_density_inputs(arena, density_inputs.clone());
    converter.already_converted_functions = already_converted_functions;

    for f in spmt_df.helper_functions.iter() {
        let (rcl_func, fconv) = convert_function(
            f,
            arena,
            converter.already_converted_functions,
            density_inputs.clone(),
        );
        rcl_funcs.push(rcl_func);
        converter.already_converted_functions = fconv.already_converted_functions;
    }

    // Register variables in converter
    for var in &spmt_df.variables {
        let rcl_var = Rc::new(rcl::Variable {
            name: var.name.as_deref().map(sanitize_name),
            t: types::convert_type(&var.t),
            mutable: true,
        });
        converter.register_variable(var.clone(), rcl_var.clone());
        rcl_func.add_variable(rcl_var);
    }

    // Convert body statements
    for stmt in &spmt_df.body {
        let converted_stmt = converter.convert_statement(stmt);
        rcl_func.add_statement(converted_stmt);
    }

    let rcl_func_ref = Interned::new(arena.alloc(rcl_func));
    // Store the converted function in the map
    converter
        .already_converted_functions
        .insert(spmt_df.addr(), rcl_func_ref.clone());

    (rcl_funcs, rcl_func_ref, converter)
}
