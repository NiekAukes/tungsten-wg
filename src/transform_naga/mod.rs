/*
Conversion from SPMT (Single Program Multi Target) to Naga IR.
This module transforms SPMT density functions into GPU shader representations
using the Naga intermediate representation, which can then be compiled to
WGSL, SPIR-V, or other GPU shader languages.

The transformation is organized into focused sub-modules:
- types: Manages type conversions between SPMT and Naga (with configurable precision)
- expression: Converts SPMT expressions to Naga expressions
- statement: Converts SPMT statements to Naga statements
- function: Converts SPMT functions/density functions to Naga functions
- extern_functions: Generates Naga function declarations for extern calls
*/

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use naga::{Expression, Function, GlobalVariable, Handle, Module};

use crate::orchestrate::Scale;
use crate::spmt::model::{self as spmt, Addr, DensityFunctionRef, DensityInput};
use crate::transform_naga::extern_functions::ExternFunctionConverter;

pub mod expression;
pub mod extern_functions;
pub mod function;
pub mod statement;
pub mod types;

pub use types::Precision;

/// Key for identifying density function inputs, matching the RCL convention.
/// Uniquely identifies a density function call by its function pointer, dimensions, and origin scale.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InputKey {
    Density {
        density_function: *const (),
        dimensions: (i32, i32, i32),
        scaled_origin: Scale,
    },
    UnnamedVar {
        key: *const (),
    },
    NamedVar {
        name: String,
    },
}

impl<'m> InputKey {
    pub fn new(
        density_function: DensityFunctionRef<'m>,
        dimensions: (i32, i32, i32),
        scaled_origin: Scale,
    ) -> Self {
        InputKey::Density {
            density_function: density_function.addr(),
            dimensions,
            scaled_origin,
        }
    }
}

impl<'m> From<&DensityInput<'m>> for InputKey {
    fn from(input: &DensityInput<'m>) -> Self {
        InputKey::Density {
            density_function: input.density_function.addr(),
            dimensions: input.dimensions,
            scaled_origin: Scale::from(input.scaled_origin),
        }
    }
}

impl<'m> From<spmt::Var<'m>> for InputKey {
    fn from(var: spmt::Var<'m>) -> Self {
        if let Some(name) = &var.name {
            InputKey::NamedVar { name: name.clone() }
        } else {
            InputKey::UnnamedVar { key: var.addr() }
        }
    }
}

/// Tracks a density function argument: its parameter index and array length.
#[derive(Debug, Clone)]
pub struct DensityArgInfo {
    /// Global variable handle of the packed density input struct.
    pub variable: Handle<GlobalVariable>,
    /// Member index inside the packed density input struct.
    pub member_index: u32,
    /// The naga type handle for the array type.
    pub array_ty: Handle<naga::Type>,
}

/// Tracks a permutation table argument: its parameter index.
#[derive(Debug, Clone)]
pub struct PermTableArgInfo {
    /// Global variable handle of the packed permutation table struct.
    pub variable: Handle<GlobalVariable>,
    /// Member index inside the packed permutation table struct.
    pub member_index: u32,
}

/// Converter state for transforming a single SPMT function to Naga IR.
///
/// Holds the mapping from SPMT variables to Naga expression handles,
/// along with the function's expression arena and body statements.
#[derive(Debug)]
pub struct NagaFunctionConverter<'m> {
    /// Maps SPMT variable keys to Naga local variable expression handles (pointers).
    var_map: HashMap<InputKey, Handle<Expression>>,
    /// Set of keys in var_map that are direct values (e.g. function arguments) rather than pointers.
    /// These should NOT be wrapped in Expression::Load.
    value_vars: HashSet<InputKey>,
    /// Maps density function input keys to their argument info.
    density_arg_map: HashMap<InputKey, DensityArgInfo>,
    /// Maps permutation table names to their argument info.
    perm_table_arg_map: HashMap<String, PermTableArgInfo>,
    /// Counter for generating unique function names.
    function_counter: usize,
    /// Cache of already-converted SPMT functions (key: SPMT pointer addr → Naga function handle).
    pub already_converted_functions: HashMap<*const (), Handle<Function>>,
    pub extern_converter: &'m ExternFunctionConverter<'m>,
    /// Total number of arguments registered for the current function.
    arg_count: u32,
}

const HELPER_MODULE_WGSL: &str = include_str!("helpers/helpers.wgsl");

impl<'a> NagaFunctionConverter<'a> {
    /// Create a converter initialized with previous conversion state (for chaining).
    pub fn with_state(
        already_converted: HashMap<*const (), Handle<Function>>,
        extern_converter: &'a ExternFunctionConverter<'a>,
    ) -> Self {
        NagaFunctionConverter {
            var_map: HashMap::new(),
            value_vars: HashSet::new(),
            density_arg_map: HashMap::new(),
            perm_table_arg_map: HashMap::new(),
            function_counter: 0,
            already_converted_functions: already_converted,
            arg_count: 0,
            extern_converter,
        }
    }

    pub fn derive_new_with_state(&self) -> Self {
        NagaFunctionConverter {
            var_map: HashMap::new(),
            value_vars: HashSet::new(),
            density_arg_map: self.density_arg_map.clone(),
            perm_table_arg_map: self.perm_table_arg_map.clone(),
            function_counter: self.function_counter,
            already_converted_functions: self.already_converted_functions.clone(),
            arg_count: 0,
            extern_converter: self.extern_converter,
        }
    }

    /// Register a density input argument. Returns the argument index.
    pub fn register_density_arg(
        &mut self,
        key: InputKey,
        result: Handle<GlobalVariable>,
        member_index: u32,
        array_ty: Handle<naga::Type>,
    ) {
        // let idx = self.arg_count;
        // self.density_arg_map.insert(
        //     key,
        //     DensityArgInfo {
        //         arg_index: idx,
        //         array_ty,
        //     },
        // );
        // self.arg_count += 1;
        // idx
        self.density_arg_map.insert(
            key,
            DensityArgInfo {
                variable: result,
                member_index,
                array_ty,
            },
        );
    }

    /// Register a permutation table argument. Returns the argument index.
    pub fn register_perm_table_arg(
        &mut self,
        handle: Handle<GlobalVariable>,
        member_index: u32,
        name: String,
    ) {
        // let idx = self.arg_count;
        // self.perm_table_arg_map
        //     .insert(name, PermTableArgInfo { arg_index: idx });
        // self.arg_count += 1;
        // idx
        self.perm_table_arg_map.insert(
            name,
            PermTableArgInfo {
                variable: handle,
                member_index,
            },
        );
    }

    /// Register a positional argument (pos3, origin, etc.). Returns the argument index.
    pub fn register_positional_arg(&mut self) -> u32 {
        let idx = self.arg_count;
        self.arg_count += 1;
        idx
    }

    /// Generate a unique function name for unnamed functions.
    pub fn next_function_name(&mut self) -> String {
        let name = format!("func_{}", self.function_counter);
        self.function_counter += 1;
        name
    }
}

/// Convert an entire SPMT program to separate Naga modules.
///
/// Each density function becomes its own Naga module (and thus its own shader).
/// Helper functions and extern calls referenced by each density function are
/// included in that density function's module.
pub fn convert_spmt_to_naga(
    program: &spmt::SPMT<'_>,
    precision: Precision,
    helpers: naga::Module,
) -> Vec<(String, Rc<RefCell<naga::Module>>)> {
    let mut results = Vec::new();

    for (i, density_function) in program.density_functions.iter().enumerate() {
        let module = Rc::new(RefCell::new(naga::Module::default()));

        let mut extern_converter = ExternFunctionConverter::new(&helpers);
        let type_cache =
            types::TypeCache::register(module.borrow_mut(), precision, &mut extern_converter);

        let mut already_converted: HashMap<*const (), Handle<Function>> = HashMap::new();

        function::convert_density_function(
            density_function,
            module.clone(),
            &type_cache,
            &mut already_converted,
            &extern_converter,
        );

        let name = density_function
            .canonical_name
            .as_deref()
            .map(types::sanitize_name)
            .unwrap_or_else(|| format!("density_{}", i));

        results.push((name, module));
    }

    results
}

pub fn parse_helpers() -> naga::Module {
    naga::front::wgsl::parse_str(HELPER_MODULE_WGSL).expect("Helper WGSL failed to parse")
}
