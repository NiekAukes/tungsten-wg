/*
Function conversion from SPMT to Naga IR.
Handles conversion of:
- SPMT helper functions (Function) → Naga functions
- SPMT density functions (DensityFunction) → Naga functions with density-specific arguments
*/

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use naga::{
    Binding, BuiltIn, Expression, Function, FunctionArgument, FunctionResult, GlobalVariable,
    Handle, LocalVariable, MemoryDecorations, ResourceBinding, Span, StorageAccess,
};

use super::expression::ExprContext;
use super::statement;
use super::types::{TypeCache, permutation_table_var_name, sanitize_name};
use super::{InputKey, NagaFunctionConverter};
use crate::orchestrate::Flatten;
use crate::spmt::model::{self as spmt, Addr};
use crate::transform_naga::extern_functions::ExternFunctionConverter;

/// Convert an SPMT helper function to a Naga function.
/// Returns the handle to the newly created function in the module.
pub fn convert_function<'a, 'm, 'b>(
    spmt_func: &spmt::Function<'a>,
    module: Rc<RefCell<naga::Module>>,
    type_cache: &'m TypeCache,
    converter: &'b mut NagaFunctionConverter<'m>,
) -> Handle<Function> {
    // Check if already converted
    if let Some(&h) = converter.already_converted_functions.get(&spmt_func.addr()) {
        return h;
    }

    let func_name = spmt_func
        .canonical_name
        .as_deref()
        .map(sanitize_name)
        .unwrap_or_else(|| converter.next_function_name());

    let mut naga_func = Function::default();
    naga_func.name = Some(func_name);
    naga_func.result = Some(FunctionResult {
        ty: type_cache.float_ty,
        binding: None,
    });

    // Reset converter state for this function
    if converter.arg_count != 0 || !converter.var_map.is_empty() {
        panic!(
            "Warning: Converter state not empty at start of function conversion. Resetting state."
        );
    }
    converter.arg_count = 0;

    // Add parameters as function arguments
    for (i, param) in spmt_func.parameters.iter().enumerate() {
        let param_ty = type_cache.convert_type(&param.t);
        let param_name = param
            .name
            .as_deref()
            .map(sanitize_name)
            .unwrap_or_else(|| format!("param{}", i));

        naga_func.arguments.push(FunctionArgument {
            name: Some(param_name),
            ty: param_ty,
            binding: None,
        });

        let arg_idx = converter.register_positional_arg();
        // Create a FunctionArgument expression so the var_map can reference it
        let arg_expr = naga_func
            .expressions
            .append(Expression::FunctionArgument(arg_idx), Span::UNDEFINED);
        let key = InputKey::from(*param);
        converter.var_map.insert(key.clone(), arg_expr);
        converter.value_vars.insert(key);
    }

    // Register local variables
    for var in &spmt_func.variables {
        let var_ty = type_cache.convert_type(&var.t);
        let local_handle = naga_func.local_variables.append(
            LocalVariable {
                name: var.name.as_deref().map(sanitize_name),
                ty: var_ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        let ptr_expr = naga_func
            .expressions
            .append(Expression::LocalVariable(local_handle), Span::UNDEFINED);
        converter.var_map.insert(InputKey::from(*var), ptr_expr);
    }

    // Convert body statements
    {
        let mut ctx: ExprContext<'m, 'a, '_> =
            ExprContext::new(&mut naga_func, module.clone(), type_cache, converter);

        for stmt in &spmt_func.body {
            statement::convert_statement(stmt, &mut ctx);
        }
    }

    // Add function to module (must appear before its callers)
    let func_handle = module
        .borrow_mut()
        .functions
        .append(naga_func, Span::UNDEFINED);
    converter
        .already_converted_functions
        .insert(spmt_func.addr(), func_handle);

    func_handle
}

/// Convert an SPMT density function to a Naga function.
/// Density functions have a specific signature:
///   fn density(pos3: vec3<i32>, origin: vec3<f32>, density_inputs..., perm_tables...) -> f32
///
/// Also converts any helper functions referenced by the density function.
pub fn convert_density_function<'a, 'b>(
    spmt_df: &spmt::DensityFunction<'a>,
    module: Rc<RefCell<naga::Module>>,
    type_cache: &TypeCache,
    already_converted: &mut HashMap<*const (), Handle<Function>>,
    extern_converter: &'b ExternFunctionConverter<'b>,
) {
    let func_name = spmt_df
        .canonical_name
        .as_deref()
        .map(sanitize_name)
        .unwrap_or_else(|| format!("density_{}", spmt_df.addr() as usize));

    let mut naga_func = Function::default();
    naga_func.name = Some(func_name);

    let mut converter =
        NagaFunctionConverter::with_state(already_converted.clone(), extern_converter);

    // === Add standard positional arguments ===

    // arg 0: pos3: vec3<i32>
    naga_func.arguments.push(FunctionArgument {
        name: Some("pos3".into()),
        ty: type_cache.vec3u_ty,
        binding: Some(Binding::BuiltIn(BuiltIn::GlobalInvocationId)),
        //binding: None,
    });
    let pos3_idx = converter.register_positional_arg();

    // arg 1: origin: vec3<f32/f64>
    // naga_func.arguments.push(FunctionArgument {
    //     name: Some("origin".into()),
    //     ty: type_cache.vec3f_ty,
    //     binding: None,
    // });
    //let origin_idx = converter.register_positional_arg();

    //@group(0) @binding(2)
    let origin_handle = module.borrow_mut().global_variables.append(
        naga::GlobalVariable {
            name: Some("origin".into()),
            space: naga::AddressSpace::Uniform,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: type_cache.vec3f_ty,
            init: None,
            memory_decorations: MemoryDecorations::default(),
        },
        Span::UNDEFINED,
    );

    let output_handle = make_output_buffer(module.clone(), type_cache, 1);

    // make the dimensions uniform as well
    let dimensions_handle = module.borrow_mut().global_variables.append(
        naga::GlobalVariable {
            name: Some("dimensions".into()),
            space: naga::AddressSpace::Uniform,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 2,
            }),
            ty: type_cache.vec3u_ty,
            init: None,
            memory_decorations: MemoryDecorations::default(),
        },
        Span::UNDEFINED,
    );

    let mut bind_counter = 3; // Start after origin's binding

    // === Add density input arguments ===
    for input in &spmt_df.density_inputs {
        let array_len = ensure_alignment(input.dimensions.flatten());
        let array_ty =
            type_cache.make_density_array_type(&mut module.borrow_mut().types, array_len);
        let fname = match &input.density_function.canonical_name {
            Some(name) => name.clone(),
            None => format!("input_{}", input.density_function.addr() as usize),
        };
        let param_name = sanitize_name(&format!("input_{}", fname));

        // naga_func.arguments.push(FunctionArgument {
        //     name: Some(param_name),
        //     ty: array_ty,
        //     binding: None,
        // });

        // let key = InputKey::from(input);
        // converter.register_density_arg(key, array_ty);

        // add global variable for this density input
        let handle = module.borrow_mut().global_variables.append(
            naga::GlobalVariable {
                name: Some(param_name.clone()),
                space: naga::AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: bind_counter,
                }),
                ty: array_ty,
                init: None,
                memory_decorations: MemoryDecorations::default(),
            },
            Span::UNDEFINED,
        );
        bind_counter += 1;

        let key = InputKey::from(input);
        converter.register_density_arg(key, handle, array_ty);
    }

    // === Add permutation table arguments ===
    for input in &spmt_df.permutation_table_inputs {
        let param_name = permutation_table_var_name(input);
        // naga_func.arguments.push(FunctionArgument {
        //     name: Some(param_name.clone()),
        //     ty: type_cache.perm_table_ty,
        //     binding: None,
        // });
        // converter.register_perm_table_arg(param_name);
        // add global variable for this permutation table
        let handle = module.borrow_mut().global_variables.append(
            naga::GlobalVariable {
                name: Some(param_name.clone()),
                space: naga::AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: bind_counter,
                }),
                ty: type_cache.perm_table_ty,
                init: None,
                memory_decorations: MemoryDecorations::default(),
            },
            Span::UNDEFINED,
        );
        bind_counter += 1;
        converter.register_perm_table_arg(handle, param_name);
    }

    // === Register pos3 and origin argument expressions in var_map ===
    {
        // let pos3_expr = naga_func
        //     .expressions
        //     .append(Expression::FunctionArgument(pos3_idx), Span::UNDEFINED);
        let pos3_expr = naga_func
            .expressions
            .append(Expression::FunctionArgument(pos3_idx), Span::UNDEFINED);
        converter.var_map.insert(
            InputKey::NamedVar {
                name: "pos3".into(),
            },
            pos3_expr,
        );

        converter.value_vars.insert(InputKey::NamedVar {
            name: "pos3".into(),
        });

        let origin_expr = naga_func
            .expressions
            .append(Expression::GlobalVariable(origin_handle), Span::UNDEFINED);

        let origin_load = naga_func.expressions.append(
            Expression::Load {
                pointer: origin_expr,
            },
            Span::UNDEFINED,
        );
        converter.var_map.insert(
            InputKey::NamedVar {
                name: "origin".into(),
            },
            origin_load,
        );
        converter.value_vars.insert(InputKey::NamedVar {
            name: "origin".into(),
        });
    }

    // Also check helper functions' parameters for pos3/origin that might be
    // referenced in the density function's body via closures.
    // (Helper functions bring pos3/origin as free variables from the parent scope.)

    // === Convert helper functions first ===
    for helper in &spmt_df.helper_functions {
        if !converter
            .already_converted_functions
            .contains_key(&helper.addr())
        {
            convert_function(
                helper,
                module.clone(),
                type_cache,
                &mut converter.derive_new_with_state(),
            );
        }
    }

    // === Register local variables (skip pos3 and origin, already mapped above) ===
    for var in &spmt_df.variables {
        if converter.var_map.contains_key(&InputKey::from(*var)) {
            continue; // already registered as function argument
        }
        let var_ty = type_cache.convert_type(&var.t);
        let local_handle = naga_func.local_variables.append(
            LocalVariable {
                name: var.name.as_deref().map(sanitize_name),
                ty: var_ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        let ptr_expr = naga_func
            .expressions
            .append(Expression::LocalVariable(local_handle), Span::UNDEFINED);
        converter.var_map.insert(InputKey::from(*var), ptr_expr);
    }

    // === Convert body statements ===
    {
        let mut ctx = ExprContext::new(&mut naga_func, module.clone(), type_cache, &mut converter);
        for stmt in &spmt_df.body {
            if matches!(stmt, spmt::Statement::Return(_)) {
                statement::convert_density_return_statement(
                    stmt,
                    output_handle,
                    pos3_idx,
                    dimensions_handle,
                    &mut ctx,
                );
            } else {
                statement::convert_statement(stmt, &mut ctx);
            }
        }
    }

    module.borrow_mut().entry_points.push(naga::EntryPoint {
        name: naga_func.name.clone().unwrap_or_else(|| "main".into()),
        stage: naga::ShaderStage::Compute,
        function: naga_func,
        workgroup_size: [16, 1, 16],
        early_depth_test: None,
        workgroup_size_overrides: None,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    // === Add function to module ===
    // let func_handle = module.functions.append(naga_func, Span::UNDEFINED);
    // already_converted.insert(spmt_df.addr(), func_handle);

    // Merge converter state back
    already_converted.extend(converter.already_converted_functions.into_iter());
}

fn ensure_alignment(len: usize) -> u32 {
    // Permutation tables need to be aligned to 16 bytes for efficient access as vec3<f32>.
    // If the length is not a multiple of 4, pad it to the next multiple of 4.
    let l = len as u32;
    if l % 4 == 0 { l } else { l + (4 - (l % 4)) }
}

fn make_output_buffer(
    module: Rc<RefCell<naga::Module>>,
    type_cache: &TypeCache,
    binding: u32,
) -> Handle<GlobalVariable> {
    let param_name = "output_buffer";
    module.borrow_mut().global_variables.append(
        naga::GlobalVariable {
            name: Some(param_name.into()),
            space: naga::AddressSpace::Storage {
                access: StorageAccess::STORE.union(StorageAccess::LOAD),
            },
            binding: Some(ResourceBinding { group: 0, binding }),
            ty: type_cache.output_ty,
            init: None,
            memory_decorations: MemoryDecorations::default(),
        },
        Span::UNDEFINED,
    )
}
