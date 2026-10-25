use std::collections::HashSet;

use tungsten_ir::model::{ComputeUnit, HostInput, OptimizedProgram, PermutationTableInput, Ref};

/// Replaces characters that are not valid in generated identifiers.
pub fn sanitize_name(name: &str) -> String {
    name.replace(':', "_")
        .replace('/', "_")
        .replace('<', "_")
        .replace('>', "_")
        .replace('-', "_")
}

/// `base`, or `base_N` for the first free N if `base` is already taken.
pub fn unique(used: &mut HashSet<String>, base: String) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    let mut n = 1;
    loop {
        let candidate = format!("{}_{}", base, n);
        if used.insert(candidate.clone()) {
            return candidate;
        }
        n += 1;
    }
}

/// Field name of a permutation table in `AutoConsts`.
pub fn auto_const_field_name(table: &PermutationTableInput) -> String {
    match table {
        PermutationTableInput::PerlinNoise { ident, subident, subident_index } => sanitize_name(
            &format!("{}_{}_{}", ident, subident_index, subident.as_deref().unwrap_or("")),
        ),
        PermutationTableInput::Base3DNoise => "base3d_perm_table".to_string(),
    }
}

/// Field name of a static host input in `HostConsts`.
pub fn host_field_name(input: &HostInput) -> String {
    input.get_name()
}

pub fn unit_base_name(program: &OptimizedProgram, cu: Ref<ComputeUnit>) -> String {
    let ComputeUnit::DensityFunction { function, source_hash, .. } = &program.compute_units[cu];
    program.functions[*function]
        .canonical_name
        .as_deref()
        .map(sanitize_name)
        .unwrap_or_else(|| format!("density_function_{:x}", source_hash))
}