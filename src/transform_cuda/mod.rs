/*
Conversion from SPMT to CUDA C++.
This module provides utilities to translate SPMT density functions into
CUDA kernel code suitable for GPU execution.

The transformation is organized into focused sub-modules:
- types:      SPMT → CUDA type / operator conversions
- expression: Converts SPMT expressions to CUDA expressions
- statement:  Converts SPMT statements to CUDA statements
- function:   Converts SPMT functions to CUDA functions / kernels
*/

use std::collections::HashMap;
use std::rc::Rc;

use crate::cuda::model as cuda;
use crate::orchestrate::Scale;
use crate::spmt::model::{self as spmt, Addr, DensityFunctionRef, DensityInput};

pub mod expression;
pub mod function;
pub mod statement;
pub mod types;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const PERM_TABLES_STRUCT_NAME: &str = "PermutationTables";

// ---------------------------------------------------------------------------
// InputKey — identical role to transform_rcl::InputKey
// ---------------------------------------------------------------------------

/// Uniquely identifies one density-function input dependency (its function
/// pointer + sampling dimensions + scaled origin).  Used as the key for
/// parameter maps so that the same input used at different scales can be
/// distinguished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InputKey {
    density_function: *const (),
    dimensions: (i32, i32, i32),
    scaled_origin: Scale,
}

impl<'m> InputKey {
    pub fn new(
        density_function: DensityFunctionRef<'m>,
        dimensions: (i32, i32, i32),
        scaled_origin: Scale,
    ) -> Self {
        InputKey {
            density_function: density_function.addr(),
            dimensions,
            scaled_origin,
        }
    }
}

impl<'m> From<&DensityInput<'m>> for InputKey {
    fn from(input: &DensityInput<'m>) -> Self {
        InputKey {
            density_function: input.density_function.addr(),
            dimensions: input.dimensions,
            scaled_origin: Scale::from(input.scaled_origin),
        }
    }
}

impl<'m> From<spmt::Var<'m>> for InputKey {
    fn from(var: spmt::Var<'m>) -> Self {
        InputKey {
            density_function: var.addr(),
            dimensions: (0, 0, 0),
            scaled_origin: Scale::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// CudaFunctionConverter
// ---------------------------------------------------------------------------

/// State carried while converting a single SPMT density function (or helper)
/// into CUDA.
pub struct CudaFunctionConverter<'m> {
    /// Maps SPMT variable addresses → CUDA `Rc<Variable>`.
    var_map: HashMap<InputKey, Rc<cuda::Variable>>,
    /// Memoised: SPMT function pointer → already-converted CUDA function ref.
    pub already_converted_functions: HashMap<*const (), cuda::FunctionRef<'m>>,
    pub arena: &'m bumpalo::Bump,
    /// The density inputs that were registered as function parameters.
    pub density_function_inputs: Rc<HashMap<InputKey, cuda::Parameter>>,
}

impl<'m> CudaFunctionConverter<'m> {
    pub fn new(arena: &'m bumpalo::Bump) -> Self {
        CudaFunctionConverter {
            var_map: HashMap::new(),
            already_converted_functions: HashMap::new(),
            arena,
            density_function_inputs: Rc::new(HashMap::new()),
        }
    }

    pub fn new_with_density_inputs(
        arena: &'m bumpalo::Bump,
        density_function_inputs: Rc<HashMap<InputKey, cuda::Parameter>>,
    ) -> Self {
        // Pre-populate var_map so that density-input variables resolve immediately.
        let mut var_map = HashMap::new();
        for (key, param) in density_function_inputs.as_ref() {
            let cuda_var = Rc::new(cuda::Variable {
                name: Some(param.name.clone()),
                t: param.t.clone(),
                memory_qualifier: None,
            });
            var_map.insert(*key, cuda_var);
        }
        CudaFunctionConverter {
            var_map,
            already_converted_functions: HashMap::new(),
            arena,
            density_function_inputs,
        }
    }

    // -----------------------------------------------------------------------
    // Variable helpers
    // -----------------------------------------------------------------------

    /// Register an SPMT variable → CUDA variable mapping.
    pub fn register_variable(&mut self, spmt_var: spmt::Var<'_>, cuda_var: Rc<cuda::Variable>) {
        self.var_map.insert(InputKey::from(spmt_var), cuda_var);
    }

    /// Return the CUDA variable for an SPMT var, creating a fresh one if not seen.
    pub fn get_or_create_variable(&mut self, spmt_var: spmt::Var<'_>) -> Rc<cuda::Variable> {
        let key = InputKey::from(spmt_var);
        if let Some(v) = self.var_map.get(&key) {
            return v.clone();
        }
        // Create a variable with the correct name and type from the SPMT var.
        let var = Rc::new(cuda::Variable {
            name: spmt_var.name.as_deref().map(crate::transform_cuda::sanitize_name),
            t: crate::transform_cuda::types::convert_type(&spmt_var.t),
            memory_qualifier: None,
        });
        self.var_map.insert(key, var.clone());
        var
    }

    /// Look up an existing variable without creating one.
    pub fn get_variable(&self, key: &InputKey) -> Option<Rc<cuda::Variable>> {
        self.var_map.get(key).cloned()
    }
}

// ---------------------------------------------------------------------------
// Name sanitisation
// ---------------------------------------------------------------------------

/// Replace characters illegal in C++ identifiers.
pub fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

// ---------------------------------------------------------------------------
// Public entry-point
// ---------------------------------------------------------------------------

/// Add a SPMT density function (and all its helper functions) to a
/// `CudaModule` as a `__global__` kernel + `__device__` helpers.
///
/// Returns the converter state (which contains `already_converted_functions`)
/// so the caller can pass it to subsequent calls, mirroring the RCL API.
pub fn add_density_to_cuda_module<'a, 'm>(
    cuda_module: &mut cuda::CudaModule<'m>,
    spmt_df: &'a spmt::DensityFunction<'a>,
    arena: &'m bumpalo::Bump,
    already_converted_functions: HashMap<*const (), cuda::FunctionRef<'m>>,
) -> CudaFunctionConverter<'m> {
    let (device_funcs, kernel, converter) =
        function::convert_density_function(spmt_df, arena, already_converted_functions);

    for f in device_funcs {
        cuda_module.add_device_function(f);
    }
    cuda_module.add_kernel(kernel);
    converter
}
