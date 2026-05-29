/*
Function conversion from SPMT to CUDA C++.
Handles conversion of helper function definitions and density-function kernels.
*/

use std::collections::HashMap;
use std::rc::Rc;

use super::{CudaFunctionConverter, InputKey, sanitize_name};
use crate::cuda::model as cuda;
use crate::spmt::model::{self as spmt, Addr, Interned};
use crate::transform_cuda::types::{convert_type, permutation_table_param_name};

// ---------------------------------------------------------------------------
// Helper function  (spmt::Function → __device__ cuda::CudaFunction)
// ---------------------------------------------------------------------------

/// Convert a SPMT helper function to a `__device__` CUDA function.
///
/// Returns the converted ref and the converter state (which carries the
/// `already_converted_functions` memo-table for use by the caller).
pub fn convert_function<'a, 'm>(
    spmt_func: &spmt::Function<'a>,
    arena: &'m bumpalo::Bump,
    already_converted_functions: HashMap<*const (), cuda::FunctionRef<'m>>,
    density_inputs: Rc<HashMap<InputKey, cuda::Parameter>>,
) -> (cuda::FunctionRef<'m>, CudaFunctionConverter<'m>) {
    let mut converter =
        CudaFunctionConverter::new_with_density_inputs(arena, density_inputs.clone());
    converter.already_converted_functions = already_converted_functions;

    let mut cuda_func = cuda::CudaFunction::new(
        cuda::FunctionQualifier::Device,
        spmt_func
            .canonical_name
            .as_deref()
            .map(sanitize_name),
        cuda::Type::Float,
    );

    // Parameters from the SPMT function signature.
    for (i, param_var) in spmt_func.parameters.iter().enumerate() {
        let param_type = convert_type(&param_var.t);
        let param_name = sanitize_name(
            &param_var.name.clone().unwrap_or_else(|| format!("param{}", i)),
        );
        cuda_func.add_parameter(param_name, param_type, false);
    }

    // Also pass density-input arrays through.
    for (_, param) in density_inputs.as_ref() {
        cuda_func.add_parameter(param.name.clone(), param.t.clone(), param.is_const);
    }

    // Register local variables.
    for var in &spmt_func.variables {
        let cuda_var = Rc::new(cuda::Variable {
            name: var.name.as_deref().map(sanitize_name),
            t: convert_type(&var.t),
            memory_qualifier: None,
        });
        converter.register_variable(var.clone(), cuda_var.clone());
        cuda_func.add_variable(cuda_var);
    }

    // Body.
    for stmt in &spmt_func.body {
        let converted = converter.convert_statement(stmt);
        cuda_func.add_statement(converted);
    }

    let func_ref = Interned::new(arena.alloc(cuda_func));
    converter
        .already_converted_functions
        .insert(spmt_func.addr(), func_ref);

    (func_ref, converter)
}

// ---------------------------------------------------------------------------
// Density function  (spmt::DensityFunction → __global__ kernel)
// ---------------------------------------------------------------------------

/// Convert a SPMT density function to a `__global__` CUDA kernel, plus any
/// `__device__` helper functions it uses.
///
/// Returns:
///   - `Vec<FunctionRef>`: `__device__` helpers (in dependency order)
///   - `FunctionRef`:      the `__global__` kernel
///   - `CudaFunctionConverter`: converter state for continuing the conversion
pub fn convert_density_function<'a, 'm>(
    spmt_df: &'a spmt::DensityFunction<'a>,
    arena: &'m bumpalo::Bump,
    already_converted_functions: HashMap<*const (), cuda::FunctionRef<'m>>,
) -> (
    Vec<cuda::FunctionRef<'m>>,
    cuda::FunctionRef<'m>,
    CudaFunctionConverter<'m>,
) {
    let mut device_funcs: Vec<cuda::FunctionRef<'m>> = Vec::new();

    // ── Build the density-input parameter map ────────────────────────────
    let mut density_input_params: HashMap<InputKey, cuda::Parameter> = HashMap::new();
    for (i, input) in spmt_df.density_inputs.iter().enumerate() {
        let param_name = sanitize_name(
            &input
                .var
                .name
                .clone()
                .unwrap_or_else(|| format!("input_{}", i)),
        );
        let param = cuda::Parameter {
            name: param_name.clone(),
            t: cuda::Type::ConstPointer(Box::new(cuda::Type::Float)),
            is_const: true,
        };
        density_input_params.insert(InputKey::from(input), param);
    }
    let density_inputs = Rc::new(density_input_params);

    // ── Converter ────────────────────────────────────────────────────────
    let mut converter =
        CudaFunctionConverter::new_with_density_inputs(arena, density_inputs.clone());
    converter.already_converted_functions = already_converted_functions;

    // ── Convert helper (__device__) functions ────────────────────────────
    for helper in &spmt_df.helper_functions {
        let (func_ref, fconv) = convert_function(
            helper,
            arena,
            converter.already_converted_functions.clone(),
            density_inputs.clone(),
        );
        device_funcs.push(func_ref);
        // Merge memoisation tables.
        converter
            .already_converted_functions
            .extend(fconv.already_converted_functions.into_iter());
    }

    // ── Build the __global__ kernel ──────────────────────────────────────
    let kernel_name = spmt_df
        .canonical_name
        .as_deref()
        .map(sanitize_name);

    let mut kernel = cuda::CudaFunction::new(
        cuda::FunctionQualifier::Global,
        kernel_name,
        cuda::Type::Void,
    );

    // Standard parameters:
    //   int3 base_pos    — bottom-left voxel coordinate of this batch
    //   int3 dimensions  — size of the output volume (x * y * z threads)
    //   float3 origin    — world-space origin offset (passed into the body as `origin`)
    kernel.add_parameter("base_pos".to_string(), cuda::Type::Struct("int3".to_string()), false);
    kernel.add_parameter("dimensions".to_string(), cuda::Type::Struct("int3".to_string()), false);
    kernel.add_parameter("origin".to_string(), cuda::Type::Struct("float3".to_string()), false);

    // Density input pointers: `const double* input_N`
    for (_, param) in density_inputs.as_ref() {
        kernel.add_parameter(param.name.clone(), param.t.clone(), param.is_const);
    }

    // Permutation table pointers: `const int8_t* perm_table_X`
    for perm in &spmt_df.permutation_table_inputs {
        let name = permutation_table_param_name(perm);
        kernel.add_parameter(
            name,
            cuda::Type::ConstPointer(Box::new(cuda::Type::Int8)),
            true,
        );
    }

    // Output pointer: `double* output`
    kernel.add_parameter(
        "output".to_string(),
        cuda::Type::Pointer(Box::new(cuda::Type::Float)),
        false,
    );

    // ── Kernel body ──────────────────────────────────────────────────────
    // Use InlineCuda to compute the flat thread index and derive pos3.
    // `base_pos` and `dimensions` are kernel parameters (int3).
    kernel.add_statement(cuda::Statement::InlineCuda(
        "int tid = threadIdx.x + blockIdx.x * blockDim.x;".to_string(),
    ));
    kernel.add_statement(cuda::Statement::InlineCuda(
        "if (tid >= dimensions.x * dimensions.y * dimensions.z) return;".to_string(),
    ));
    kernel.add_statement(cuda::Statement::InlineCuda(
        "int ux__ = tid % dimensions.x;".to_string(),
    ));
    kernel.add_statement(cuda::Statement::InlineCuda(
        "int uy__ = (tid / dimensions.x) % dimensions.y;".to_string(),
    ));
    kernel.add_statement(cuda::Statement::InlineCuda(
        "int uz__ = tid / (dimensions.x * dimensions.y);".to_string(),
    ));
    kernel.add_statement(cuda::Statement::InlineCuda(
        "int3 pos3 = make_int3(base_pos.x + ux__, base_pos.y + uy__, base_pos.z + uz__);".to_string(),
    ));

    // Register local variables from the SPMT density function.
    for var in &spmt_df.variables {
        let cuda_var = Rc::new(cuda::Variable {
            name: var.name.as_deref().map(sanitize_name),
            t: convert_type(&var.t),
            memory_qualifier: None,
        });
        converter.register_variable(var.clone(), cuda_var.clone());
        kernel.add_variable(cuda_var);
    }

    // A `result` variable to capture the return value from the body.
    let result_var = Rc::new(cuda::Variable {
        name: Some("result".to_string()),
        t: cuda::Type::Float,
        memory_qualifier: None,
    });
    kernel.add_statement(cuda::Statement::Declare {
        variable: result_var.clone(),
        init: Some(cuda::Expression::F64Literal(0.0)),
        is_const: false,
    });

    // Convert body statements; rewrite Return(expr) → Assign(result, expr).
    for stmt in &spmt_df.body {
        let cuda_stmt = converter.convert_statement(stmt);
        let cuda_stmt = rewrite_return_to_assign(cuda_stmt, result_var.clone());
        kernel.add_statement(cuda_stmt);
    }

    // Write the result to the output buffer: `output[tid] = result;`
    kernel.add_statement(cuda::Statement::InlineCuda(
        "output[tid] = result;".to_string(),
    ));

    let kernel_ref = Interned::new(arena.alloc(kernel));
    converter
        .already_converted_functions
        .insert(spmt_df.addr(), kernel_ref);

    (device_funcs, kernel_ref, converter)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rewrite a `Return(expr)` at the top level of the body into
/// `Assign { result_var, expr }`, so that kernels (which return void)
/// can capture the computed value before writing it to the output buffer.
fn rewrite_return_to_assign<'m>(
    stmt: cuda::Statement<'m>,
    result_var: Rc<cuda::Variable>,
) -> cuda::Statement<'m> {
    match stmt {
        cuda::Statement::Return(Some(expr)) => cuda::Statement::Assign {
            target: result_var,
            value: expr,
        },
        cuda::Statement::Return(None) => cuda::Statement::Block(vec![]),
        other => other,
    }
}
