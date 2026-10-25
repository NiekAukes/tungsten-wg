use std::collections::HashSet;

use tungsten_ir::model::{ComputeUnit, HostInput, OptimizedProgram, PermutationTableInput, Ref};

/// Per-language naming: names that must never be generated, and how to make a legal identifier.
/// Each backend builds its own [`super::Interface`] with its rules, so names may differ between backends.
#[derive(Debug, Clone)]
pub struct NamingRules {
    /// Identifiers that must never be produced (keywords, runtime type names, ...).
    pub reserved: Vec<String>,
    /// Turns an arbitrary name into a legal identifier of the target language.
    pub sanitize: fn(&str) -> String,
}

const RUST_KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "type",
    "unsafe", "use", "where", "while", "abstract", "become", "box", "do", "final", "macro",
    "override", "priv", "try", "typeof", "unsized", "virtual", "yield", "gen",
];

/// Java keywords (including contextual ones), literals, and names the generated Java file
/// declares or uses unqualified.
pub const JAVA_RESERVED: &[&str] = &[
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
    "var",
    "record",
    "yield",
    "sealed",
    "permits",
    "non_sealed",
    "when",
    "module",
    "open",
    "exports",
    "requires",
    "true",
    "false",
    "null",
    "_",
    // generated and runtime types
    "HostConsts",
    "AutoConsts",
    "InternalBuffers",
    "Orchestration",
    "Config",
    "Vec3",
    "Pos3",
    "PositionIterator",
    "Helpers",
    "PerlinNoiseSampler",
    "InterpolatedNoiseSampler",
    "DoubleRef",
    "FloatRef",
    "IntRef",
    "LongRef",
    "BoolRef",
    // java.lang names used unqualified
    "Math",
    "Double",
    "Float",
    "Integer",
    "Long",
    "Boolean",
    "Object",
    "String",
    "System",
    // uniform parameters, kernel locals and Orchestration members
    "p",
    "config",
    "buffers",
    "auto_constants",
    "host_consts",
    "origin_x",
    "origin_y",
    "origin_z",
    "out",
    "idx",
    "ix",
    "iy",
    "iz",
    "create",
];

impl NamingRules {
    pub fn rust() -> Self {
        Self {
            reserved: RUST_KEYWORDS.iter().map(|s| s.to_string()).collect(),
            sanitize: sanitize_name,
        }
    }

    pub fn java() -> Self {
        Self {
            reserved: JAVA_RESERVED.iter().map(|s| s.to_string()).collect(),
            sanitize: java_sanitize,
        }
    }

    /// Adds more reserved names (e.g. the generated class name).
    pub fn with_reserved(mut self, extra: impl IntoIterator<Item = String>) -> Self {
        self.reserved.extend(extra);
        self
    }

    /// A fresh `used` set seeded with the reserved names.
    pub fn used_set(&self) -> HashSet<String> {
        self.reserved.iter().cloned().collect()
    }
}

/// A legal (ASCII) Java identifier: other characters become `_`, a leading digit gets a `_` prefix,
/// and the lone `_` (a keyword since Java 9) becomes `_0`.
pub fn java_sanitize(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    if s == "_" {
        s = "_0".to_string();
    }
    s
}

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
pub fn auto_const_field_name(rules: &NamingRules, table: &PermutationTableInput) -> String {
    match table {
        PermutationTableInput::PerlinNoise {
            ident,
            subident,
            subident_index,
        } => (rules.sanitize)(&format!(
            "{}_{}_{}",
            ident,
            subident_index,
            subident.as_deref().unwrap_or("")
        )),
        PermutationTableInput::Base3DNoise => "base3d_perm_table".to_string(),
    }
}

/// Field name of a static host input in `HostConsts`.
pub fn host_field_name(rules: &NamingRules, input: &HostInput) -> String {
    (rules.sanitize)(&input.get_name())
}

pub fn unit_base_name(
    rules: &NamingRules,
    program: &OptimizedProgram,
    cu: Ref<ComputeUnit>,
) -> String {
    let ComputeUnit::DensityFunction {
        function,
        source_hash,
        ..
    } = &program.compute_units[cu];
    program.functions[*function]
        .canonical_name
        .as_deref()
        .map(rules.sanitize)
        .unwrap_or_else(|| format!("density_function_{:x}", source_hash))
}
