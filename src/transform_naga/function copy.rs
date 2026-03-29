/*
Function conversion from SPMT to Naga IR.
Handles conversion of:
- SPMT helper functions (Function) → Naga functions
- SPMT density functions (DensityFunction) → Naga functions with density-specific arguments
*/

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::rc::Rc;

use naga::{
    Binding, Block, BuiltIn, Expression, Function, FunctionArgument, FunctionResult,
    GlobalVariable, Handle, LocalVariable, MemoryDecorations, ResourceBinding, Span, Statement,
    StorageAccess,
};

use super::expression::ExprContext;
use super::statement;
use super::types::{TypeCache, permutation_table_var_name, sanitize_name};
use super::{InputKey, NagaFunctionConverter};
use crate::orchestrate::Flatten;
use crate::spmt::model::{self as spmt, Addr};
use crate::transform_naga::extern_functions::ExternFunctionConverter;

const PERM_TABLE_STRUCT_SIZE_BYTES: u32 = 1036;
const WORKGROUP_CACHE_BUDGET_BYTES: u32 = 32 * 1024;
const WORKGROUP_THREAD_COUNT: u32 = 4 * 8 * 4;
const PERM_TABLE_LENGTH: u32 = 256;
const PERM_LOADS_PER_THREAD: u32 = PERM_TABLE_LENGTH.div_ceil(WORKGROUP_THREAD_COUNT);
const PERM_FIELD_INDEX: u32 = 0;
const ORIGIN_X_FIELD_INDEX: u32 = 1;
const ORIGIN_Y_FIELD_INDEX: u32 = 2;
const ORIGIN_Z_FIELD_INDEX: u32 = 3;

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
    });
    let pos3_idx = converter.register_positional_arg();

    // arg 1: local_invocation_index: u32
    naga_func.arguments.push(FunctionArgument {
        name: Some("local_invocation_index".into()),
        ty: type_cache.u32_ty,
        binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
    });
    let local_invocation_index_idx = converter.register_positional_arg();

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

    let mut bind_counter = 3; // Start after origin/output/dimensions

    // === Pack density inputs into a single storage binding ===
    if !spmt_df.density_inputs.is_empty() {
        let mut density_members = Vec::new();
        let mut density_member_info = Vec::new();
        let mut density_offset = 0u32;

        for input in &spmt_df.density_inputs {
            let array_len = ensure_alignment(input.dimensions.flatten());
            let array_ty =
                type_cache.make_density_array_type(&mut module.borrow_mut().types, array_len);
            let fname = match &input.density_function.canonical_name {
                Some(name) => name.clone(),
                None => format!("input_{}", input.density_function.addr() as usize),
            };
            let member_name = sanitize_name(&format!("input_{}", fname));
            let member_size = array_len * 4;

            density_members.push(naga::StructMember {
                name: Some(member_name),
                ty: array_ty,
                binding: None,
                offset: density_offset,
            });
            density_member_info.push((InputKey::from(input), array_ty));
            density_offset += member_size;
        }

        let packed_density_ty = module.borrow_mut().types.insert(
            naga::Type {
                name: Some("DensityInputs".into()),
                inner: naga::TypeInner::Struct {
                    members: density_members,
                    span: density_offset,
                },
            },
            Span::UNDEFINED,
        );

        let density_handle = module.borrow_mut().global_variables.append(
            naga::GlobalVariable {
                name: Some("density_inputs".into()),
                space: naga::AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: bind_counter,
                }),
                ty: packed_density_ty,
                init: None,
                memory_decorations: MemoryDecorations::default(),
            },
            Span::UNDEFINED,
        );
        bind_counter += 1;

        for (member_index, (key, array_ty)) in density_member_info.into_iter().enumerate() {
            converter.register_density_arg(key, density_handle, member_index as u32, array_ty);
        }
    }

    // === Pack permutation tables into a single storage binding ===
    let mut perm_storage_and_cache: Option<(Handle<GlobalVariable>, Handle<GlobalVariable>, u32)> =
        None;
    if !spmt_df.permutation_table_inputs.is_empty() {
        let mut perm_members = Vec::new();
        let mut perm_names = Vec::new();
        let mut perm_offset = 0u32;

        for input in &spmt_df.permutation_table_inputs {
            let param_name = permutation_table_var_name(input);
            perm_members.push(naga::StructMember {
                name: Some(param_name.clone()),
                ty: type_cache.perm_table_ty,
                binding: None,
                offset: perm_offset,
            });
            perm_names.push(param_name);
            perm_offset += PERM_TABLE_STRUCT_SIZE_BYTES;
        }

        let packed_perm_ty = module.borrow_mut().types.insert(
            naga::Type {
                name: Some("PermutationTables".into()),
                inner: naga::TypeInner::Struct {
                    members: perm_members,
                    span: perm_offset,
                },
            },
            Span::UNDEFINED,
        );

        let perm_handle = module.borrow_mut().global_variables.append(
            naga::GlobalVariable {
                name: Some("perm_tables".into()),
                space: naga::AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: bind_counter,
                }),
                ty: packed_perm_ty,
                init: None,
                memory_decorations: MemoryDecorations::default(),
            },
            Span::UNDEFINED,
        );
        bind_counter += 1;

        let perm_table_count = perm_names.len() as u32;
        let required_workgroup_bytes = perm_table_count * PERM_TABLE_STRUCT_SIZE_BYTES;
        if required_workgroup_bytes > WORKGROUP_CACHE_BUDGET_BYTES {
            panic!(
                "Permutation table workgroup cache too large for configured budget: {} bytes required for {} tables, budget is {} bytes",
                required_workgroup_bytes,
                perm_table_count,
                WORKGROUP_CACHE_BUDGET_BYTES,
            );
        }

        let perm_cache_ty = module.borrow_mut().types.insert(
            naga::Type {
                name: Some("PermutationTablesWorkgroupCache".into()),
                inner: naga::TypeInner::Array {
                    base: type_cache.perm_table_ty,
                    size: naga::ArraySize::Constant(
                        NonZeroU32::new(perm_table_count)
                            .expect("permutation table cache must not be empty"),
                    ),
                    stride: PERM_TABLE_STRUCT_SIZE_BYTES,
                },
            },
            Span::UNDEFINED,
        );

        let perm_cache_handle = module.borrow_mut().global_variables.append(
            naga::GlobalVariable {
                name: Some("perm_tables_workgroup_cache".into()),
                space: naga::AddressSpace::WorkGroup,
                binding: None,
                ty: perm_cache_ty,
                init: None,
                memory_decorations: MemoryDecorations::default(),
            },
            Span::UNDEFINED,
        );

        perm_storage_and_cache = Some((perm_handle, perm_cache_handle, perm_table_count));

        for (member_index, param_name) in perm_names.into_iter().enumerate() {
            converter.register_perm_table_arg(perm_cache_handle, member_index as u32, param_name);
        }
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
        if let Some((perm_storage_handle, perm_cache_handle, perm_table_count)) =
            perm_storage_and_cache
        {
            emit_perm_table_cache_prologue(
                &mut naga_func,
                perm_storage_handle,
                perm_cache_handle,
                perm_table_count,
                local_invocation_index_idx,
            );
        }

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
        name: "main".into(),
        stage: naga::ShaderStage::Compute,
        function: naga_func,
        workgroup_size: [4, 8, 4],
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

fn emit_perm_table_cache_prologue(
    naga_func: &mut Function,
    perm_storage_handle: Handle<GlobalVariable>,
    perm_cache_handle: Handle<GlobalVariable>,
    perm_table_count: u32,
    local_invocation_index_idx: u32,
) {
    let local_invocation_index = naga_func.expressions.append(
        Expression::FunctionArgument(local_invocation_index_idx),
        Span::UNDEFINED,
    );
    for member_index in 0..perm_table_count {
        emit_perm_array_striped_load(
            naga_func,
            perm_storage_handle,
            perm_cache_handle,
            member_index,
            local_invocation_index,
        );
        emit_origin_component_load(
            naga_func,
            perm_storage_handle,
            perm_cache_handle,
            member_index,
            local_invocation_index,
            0,
            ORIGIN_X_FIELD_INDEX,
        );
        emit_origin_component_load(
            naga_func,
            perm_storage_handle,
            perm_cache_handle,
            member_index,
            local_invocation_index,
            1,
            ORIGIN_Y_FIELD_INDEX,
        );
        emit_origin_component_load(
            naga_func,
            perm_storage_handle,
            perm_cache_handle,
            member_index,
            local_invocation_index,
            2,
            ORIGIN_Z_FIELD_INDEX,
        );
    }

    naga_func.body.push(
        Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
        Span::UNDEFINED,
    );
}

fn emit_perm_array_striped_load(
    naga_func: &mut Function,
    perm_storage_handle: Handle<GlobalVariable>,
    perm_cache_handle: Handle<GlobalVariable>,
    member_index: u32,
    local_invocation_index: Handle<Expression>,
) {
    for load_index in 0..PERM_LOADS_PER_THREAD {
        let offset = load_index * WORKGROUP_THREAD_COUNT;
        let element_index = if offset == 0 {
            local_invocation_index
        } else {
            let offset_expr = naga_func.expressions.append(
                Expression::Literal(naga::Literal::U32(offset)),
                Span::UNDEFINED,
            );
            naga_func.expressions.append(
                Expression::Binary {
                    op: naga::BinaryOperator::Add,
                    left: local_invocation_index,
                    right: offset_expr,
                },
                Span::UNDEFINED,
            )
        };

        let storage_entry = append_perm_scalar_load(
            naga_func,
            perm_storage_handle,
            member_index,
            element_index,
        );
        let cache_entry_ptr = append_perm_scalar_ptr(
            naga_func,
            perm_cache_handle,
            member_index,
            element_index,
        );

        if offset + WORKGROUP_THREAD_COUNT <= PERM_TABLE_LENGTH {
            naga_func.body.push(
                Statement::Store {
                    pointer: cache_entry_ptr,
                    value: storage_entry,
                },
                Span::UNDEFINED,
            );
        } else {
            let perm_len = naga_func.expressions.append(
                Expression::Literal(naga::Literal::U32(PERM_TABLE_LENGTH)),
                Span::UNDEFINED,
            );
            let in_bounds = naga_func.expressions.append(
                Expression::Binary {
                    op: naga::BinaryOperator::Less,
                    left: element_index,
                    right: perm_len,
                },
                Span::UNDEFINED,
            );
            let mut accept = Block::new();
            accept.push(
                Statement::Store {
                    pointer: cache_entry_ptr,
                    value: storage_entry,
                },
                Span::UNDEFINED,
            );
            naga_func.body.push(
                Statement::If {
                    condition: in_bounds,
                    accept,
                    reject: Block::new(),
                },
                Span::UNDEFINED,
            );
        }
    }
}

fn emit_origin_component_load(
    naga_func: &mut Function,
    perm_storage_handle: Handle<GlobalVariable>,
    perm_cache_handle: Handle<GlobalVariable>,
    member_index: u32,
    local_invocation_index: Handle<Expression>,
    invocation_index: u32,
    field_index: u32,
) {
    let expected_invocation = naga_func.expressions.append(
        Expression::Literal(naga::Literal::U32(invocation_index)),
        Span::UNDEFINED,
    );
    let should_load = naga_func.expressions.append(
        Expression::Binary {
            op: naga::BinaryOperator::Equal,
            left: local_invocation_index,
            right: expected_invocation,
        },
        Span::UNDEFINED,
    );

    let storage_component = append_perm_struct_field_load(
        naga_func,
        perm_storage_handle,
        member_index,
        field_index,
    );
    let cache_component_ptr = append_perm_struct_field_ptr(
        naga_func,
        perm_cache_handle,
        member_index,
        field_index,
    );

    let mut accept = Block::new();
    accept.push(
        Statement::Store {
            pointer: cache_component_ptr,
            value: storage_component,
        },
        Span::UNDEFINED,
    );

    naga_func.body.push(
        Statement::If {
            condition: should_load,
            accept,
            reject: Block::new(),
        },
        Span::UNDEFINED,
    );
}

fn append_perm_scalar_load(
    naga_func: &mut Function,
    table_handle: Handle<GlobalVariable>,
    member_index: u32,
    element_index: Handle<Expression>,
) -> Handle<Expression> {
    let scalar_ptr = append_perm_scalar_ptr(naga_func, table_handle, member_index, element_index);
    naga_func
        .expressions
        .append(Expression::Load { pointer: scalar_ptr }, Span::UNDEFINED)
}

fn append_perm_scalar_ptr(
    naga_func: &mut Function,
    table_handle: Handle<GlobalVariable>,
    member_index: u32,
    element_index: Handle<Expression>,
) -> Handle<Expression> {
    let perm_field_ptr = append_perm_struct_field_ptr(
        naga_func,
        table_handle,
        member_index,
        PERM_FIELD_INDEX,
    );
    naga_func.expressions.append(
        Expression::Access {
            base: perm_field_ptr,
            index: element_index,
        },
        Span::UNDEFINED,
    )
}

fn append_perm_struct_field_load(
    naga_func: &mut Function,
    table_handle: Handle<GlobalVariable>,
    member_index: u32,
    field_index: u32,
) -> Handle<Expression> {
    let field_ptr = append_perm_struct_field_ptr(naga_func, table_handle, member_index, field_index);
    naga_func
        .expressions
        .append(Expression::Load { pointer: field_ptr }, Span::UNDEFINED)
}

fn append_perm_struct_field_ptr(
    naga_func: &mut Function,
    table_handle: Handle<GlobalVariable>,
    member_index: u32,
    field_index: u32,
) -> Handle<Expression> {
    let table_ptr = append_perm_table_ptr(naga_func, table_handle, member_index);
    naga_func.expressions.append(
        Expression::AccessIndex {
            base: table_ptr,
            index: field_index,
        },
        Span::UNDEFINED,
    )
}

fn append_perm_table_ptr(
    naga_func: &mut Function,
    table_handle: Handle<GlobalVariable>,
    member_index: u32,
) -> Handle<Expression> {
    let global_ptr = naga_func
        .expressions
        .append(Expression::GlobalVariable(table_handle), Span::UNDEFINED);
    naga_func.expressions.append(
        Expression::AccessIndex {
            base: global_ptr,
            index: member_index,
        },
        Span::UNDEFINED,
    )
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
