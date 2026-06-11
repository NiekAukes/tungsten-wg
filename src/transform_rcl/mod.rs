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

use crate::orchestrate::Scale;
use crate::rcl::model as rcl;
use crate::spmt::model::{self as spmt, Addr, DensityFunctionRef, DensityInput};

pub mod expression;
pub mod function;
pub mod statement;
pub mod types;

pub const PERM_TABLES_STRUCT_NAME: &str = "PermutationTables";
pub const PERLIN_NOISE_SAMPLER_STRUCT_NAME: &str = "PerlinNoiseSampler";
pub const BASE3D_NOISE_SAMPLER_STRUCT_NAME: &str = "InterpolatedNoiseSampler";
/// Converter state for transforming SPMT to RCL
pub struct RCLFunctionConverter<'m> {
    /// Maps SPMT variable addresses to RCL variables
    var_map: HashMap<InputKey, Rc<rcl::Variable>>,
    /// Counter for generating unique function names
    function_counter: usize,
    pub already_converted_functions: HashMap<*const (), rcl::FunctionRef<'m>>,
    arena: &'m bumpalo::Bump,
    density_function_inputs: Rc<HashMap<InputKey, rcl::Parameter>>,
    /// Cache for concrete variable names (used for anonymous variables).
    name_cache: HashMap<*const (), String>,
    /// Counter for generating unique anonymous variable names.
    anon_counter: usize,
    density_func_name: String,
}

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

impl<'m> RCLFunctionConverter<'m> {
    /// Create a new converter instance
    pub fn new(arena: &'m bumpalo::Bump, name: String) -> Self {
        RCLFunctionConverter {
            var_map: HashMap::new(),
            function_counter: 0,
            already_converted_functions: HashMap::new(),
            arena,
            density_function_inputs: Rc::new(HashMap::new()),
            name_cache: HashMap::new(),
            anon_counter: 0,
            density_func_name: name,
        }
    }

    pub fn new_with_density_inputs(
        arena: &'m bumpalo::Bump,
        density_function_inputs: Rc<HashMap<InputKey, rcl::Parameter>>,
        name: String,
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
            name_cache: HashMap::new(),
            anon_counter: 0,
            density_func_name: name,
        }
    }

    /// Get or create an RCL variable from an SPMT variable
    pub fn get_or_create_variable(&mut self, spmt_var: spmt::Var<'_>) -> Rc<rcl::Variable> {
        //let addr = spmt_var.addr();
        let key = InputKey::from(spmt_var);

        if let Some(var) = self.var_map.get(&key) {
            var.clone()
        } else {
            let rcl_var = Rc::new(rcl::Variable {
                name: Some(self.get_concrete_name(spmt_var)),
                t: types::convert_type(&spmt_var.t),
                mutable: true,
            });
            self.var_map.insert(key, rcl_var.clone());
            rcl_var
        }
    }

    pub fn get_variable(&self, addr: &InputKey) -> Option<Rc<rcl::Variable>> {
        self.var_map.get(addr).cloned()
    }

    pub fn add_raw_variable(&mut self, addr: InputKey, rcl_var: Rc<rcl::Variable>) {
        self.var_map.insert(addr, rcl_var);
    }

    /// Generate a unique function name
    pub fn generate_function_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.function_counter);
        self.function_counter += 1;
        name
    }

    /// Register a variable in the map
    pub fn register_variable(&mut self, spmt_var: spmt::Var<'_>, rcl_var: Rc<rcl::Variable>) {
        self.var_map.insert(InputKey::from(spmt_var), rcl_var);
    }

    /// Get a concrete name for a variable, handling Anonymous, Prefixed, and Named cases.
    /// This is similar to the ConcreteName trait in pretty.rs but doesn't require a Printer.
    pub fn get_concrete_name(&mut self, var: spmt::Var<'_>) -> String {
        use crate::spmt::model::Name;

        match &var.name {
            Name::Anonymous => {
                // Check cache first
                if let Some(name) = self.name_cache.get(&var.addr()) {
                    return name.clone();
                }
                // Generate new anonymous name
                let name = sanitize_name(&format!("var_{}", self.anon_counter));
                self.anon_counter += 1;
                self.name_cache.insert(var.addr(), name.clone());
                name
            }
            Name::Prefixed(prefix) => {
                // Check cache first
                if let Some(name) = self.name_cache.get(&var.addr()) {
                    return name.clone();
                }
                // Generate new prefixed name
                let name = sanitize_name(&format!("{}_{}", prefix, self.anon_counter));
                self.anon_counter += 1;
                self.name_cache.insert(var.addr(), name.clone());
                name
            }
            Name::Named(name) => sanitize_name(name),
        }
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
    let (rcl_funcs, rcl_density, converter, constants) =
        function::convert_density_function(spmt_df, arena, already_converted_functions);
    rcl_model.functions.extend(rcl_funcs.into_iter());
    rcl_model.constants.extend(constants.into_iter());

    rcl_model.main_functions.push(rcl_density);
    converter
}

pub fn convert_spmt_to_inline_rcl<'a, 'm>(
    program: &'a spmt::SPMT<'a>,
    arena: &'m bumpalo::Bump,
) -> rcl::RCL<'m> {
    let mut rcl_model = rcl::RCL::new();
    let mut already_converted_functions = HashMap::new();
    for density_function in &program.density_functions {
        let c = add_density_to_rcl_model(
            &mut rcl_model,
            density_function,
            &arena,
            already_converted_functions,
        );
        already_converted_functions = c.already_converted_functions;
    }
    rcl_model
}
