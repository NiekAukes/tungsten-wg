/*
Conversion from SPMT (Spatial Model Transform) to RCL (Rust Code Language).
This module provides utilities to translate SPMT density functions into
low-level CPU code representations suitable for compilation.

The transformation is organized into focused sub-modules:
- expression: Converts SPMT expressions to RCL expressions
- statement: Converts SPMT statements to RCL statements
- function: Converts SPMT functions to RCL functions
- types: Manages type conversions between SPMT and RCL
*/

use std::collections::HashMap;
use std::rc::Rc;

use crate::rcl::model as rcl;
use crate::spmt::model::{self as spmt, Addr, Interned};

pub mod expression;
pub mod function;
pub mod statement;
pub mod types;

/// Converter state for transforming SPMT to RCL
pub struct RCLFunctionConverter<'m> {
    /// Maps SPMT variable addresses to RCL variables
    var_map: HashMap<*const (), Rc<rcl::Variable>>,
    /// Counter for generating unique function names
    function_counter: usize,
    pub already_converted_functions: HashMap<*const (), rcl::FunctionRef<'m>>,
    arena: &'m bumpalo::Bump,
    density_function_inputs: Rc<HashMap<*const (), rcl::Parameter>>,
}

impl<'m> RCLFunctionConverter<'m> {
    /// Create a new converter instance
    pub fn new(arena: &'m bumpalo::Bump) -> Self {
        RCLFunctionConverter {
            var_map: HashMap::new(),
            function_counter: 0,
            already_converted_functions: HashMap::new(),
            arena,
            density_function_inputs: Rc::new(HashMap::new()),
        }
    }

    pub fn new_with_density_inputs(
        arena: &'m bumpalo::Bump,
        density_function_inputs: Rc<HashMap<*const (), rcl::Parameter>>,
    ) -> Self {
        let mut var_map = HashMap::new();
        for (key, param) in density_function_inputs.as_ref() {
            let rcl_var = Rc::new(rcl::Variable {
                name: Some(param.name.clone()),
                t: param.t.clone(),
                mutable: false,
            });
            var_map.insert(*key, rcl_var);
        }

        RCLFunctionConverter {
            var_map: var_map,
            function_counter: 0,
            already_converted_functions: HashMap::new(),
            arena,
            density_function_inputs,
        }
    }

    /// Get or create an RCL variable from an SPMT variable
    pub fn get_or_create_variable(&mut self, spmt_var: Rc<spmt::Variable>) -> Rc<rcl::Variable> {
        let addr = spmt_var.addr();

        if let Some(var) = self.var_map.get(&addr) {
            var.clone()
        } else {
            let rcl_var = Rc::new(rcl::Variable {
                name: spmt_var.name.as_deref().map(sanitize_name),
                t: types::convert_type(&spmt_var.t),
                mutable: true,
            });
            self.var_map.insert(addr, rcl_var.clone());
            rcl_var
        }
    }

    pub fn get_variable(&self, addr: *const ()) -> Option<Rc<rcl::Variable>> {
        self.var_map.get(&addr).cloned()
    }

    pub fn add_raw_variable(&mut self, addr: *const (), rcl_var: Rc<rcl::Variable>) {
        self.var_map.insert(addr, rcl_var);
    }

    /// Generate a unique function name
    pub fn generate_function_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.function_counter);
        self.function_counter += 1;
        name
    }

    /// Register a variable in the map
    pub fn register_variable(&mut self, spmt_var: Rc<spmt::Variable>, rcl_var: Rc<rcl::Variable>) {
        self.var_map.insert(spmt_var.addr(), rcl_var);
    }
}

/// Sanitize a name to make it a valid Rust identifier.
/// Replaces `:`, `/`, `<`, `>`, and `-` with `_`.
pub fn sanitize_name(name: &str) -> String {
    name.replace(':', "_")
        .replace('/', "_")
        .replace('<', "_")
        .replace('>', "_")
        .replace('-', "_")
}

/// Convert an SPMT function to an RCL function
// pub fn spmt_function_to_rcl<'m>(
//     spmt_func: &spmt::Function<'m>,
//     arena: &'m bumpalo::Bump,
// ) -> (rcl::Function<'m>, RCLFunctionConverter<'m>) {
//     function::convert_function(spmt_func, arena)
// }

pub fn add_density_to_rcl_model<'a, 'm>(
    rcl_model: &mut rcl::RCL<'m>,
    spmt_df: &'a spmt::DensityFunction<'a>,
    arena: &'m bumpalo::Bump,
    already_converted_functions: HashMap<*const (), rcl::FunctionRef<'m>>,
) -> RCLFunctionConverter<'m> {
    let (rcl_funcs, rcl_density, converter) =
        function::convert_density_function(spmt_df, arena, already_converted_functions);
    rcl_model.functions.extend(rcl_funcs.into_iter());

    rcl_model.main_functions.push(rcl_density);
    converter
}
