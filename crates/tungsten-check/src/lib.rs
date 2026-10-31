//! Lowers a tungsten-ir [`OptimizedProgram`] to the core IR of `tungsten-core`
//! (re-exported as [`core_ir`]). See this crate's `README.md`.
//!
//! The lowering only resolves names and normalizes shapes. Anything that can be *wrong* (type
//! annotations, field coverage, scoping, acyclicity) stays in the IR for the checker to judge.
//! [`lower`] collects every error, each with a [`Location`], and never panics, whatever refs the
//! program holds.
//!
//! Externs and records resolve strictly against a [`Registry`] (an unknown name is an error),
//! which the caller builds with [`RegistryBuilder`].

pub use tungsten_core as core_ir;

mod error;
mod lower;
mod map;
mod registry;

#[cfg(test)]
mod tests;

pub use error::{
    EntityKind, Item, Location, LowerError, LowerErrorKind, LowerErrors,
    ParameterProblem, Step,
};
pub use map::{CoreId, IdMap, LowerMap, RecordNames};
pub use registry::{Registry, RegistryBuilder, RegistryError};

use tungsten_ir::OptimizedProgram;

use crate::lower::run;

/// A lowered program and its side table.
#[derive(Debug, Clone, PartialEq)]
pub struct Lowered {
    pub program: core_ir::Program,
    pub map: LowerMap,
}

/// Lowers `p` against `registry`. On success the program satisfies
/// `tungsten_core::wf::wf_program`, and its extern and record tables are the registry's. On
/// failure, every error is returned, in walk order.
pub fn lower(p: &OptimizedProgram, registry: &Registry) -> Result<Lowered, LowerErrors> {
    let (mut program, mut map, errors) = run(p, registry);
    if !errors.is_empty() {
        return Err(LowerErrors(errors));
    }
    map.externs = registry.extern_names.clone();
    map.records = registry
        .record_names
        .iter()
        .zip(&registry.field_names)
        .map(|(name, fields)| RecordNames {
            name: name.clone(),
            fields: fields.clone(),
        })
        .collect();
    (program.externs, program.records) = registry.decls();
    Ok(Lowered { program, map })
}
