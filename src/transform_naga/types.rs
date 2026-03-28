/*
Type conversion utilities for translating SPMT types to Naga IR types.
Supports configurable precision (f32/f64) for density computations.
*/

use std::{cell::RefMut, vec};

use naga::{Handle, Module, Scalar, ScalarKind, Type, TypeInner, VectorSize};

use crate::{spmt::model as spmt, transform_naga::extern_functions::ExternFunctionConverter};

/// Precision mode for GPU shader computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F32,
    F64,
}

impl Precision {
    /// Returns the byte width for this precision's float type.
    pub fn float_width(self) -> u8 {
        match self {
            Precision::F32 => 4,
            Precision::F64 => 8,
        }
    }

    /// Returns the naga Scalar for float values at this precision.
    pub fn float_scalar(self) -> Scalar {
        Scalar {
            kind: ScalarKind::Float,
            width: self.float_width(),
        }
    }
}

/// Cached type handles to avoid re-registering common types.
#[derive(Debug)]
pub struct TypeCache {
    pub float_ty: Handle<Type>,
    pub output_ty: Handle<Type>,
    pub i32_ty: Handle<Type>,
    pub u32_ty: Handle<Type>,
    pub i64_ty: Handle<Type>,
    pub vec3f_ty: Handle<Type>,
    pub vec3i_ty: Handle<Type>,
    pub vec3u_ty: Handle<Type>,
    pub perm_array_ty: Handle<Type>,
    pub perm_table_ty: Handle<Type>,
}

impl TypeCache {
    /// Register all commonly used types in the module and cache their handles.
    pub fn register(
        mut module: RefMut<'_, Module>,
        precision: Precision,
        extern_converter: &mut ExternFunctionConverter,
    ) -> Self {
        let types = &mut module.types;
        let float_scalar = precision.float_scalar();

        let float_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(float_scalar),
            },
            naga::Span::UNDEFINED,
        );

        let i32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::I32),
            },
            naga::Span::UNDEFINED,
        );

        let i64_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::I64),
            },
            naga::Span::UNDEFINED,
        );

        let vec3f_ty = types.insert(
            Type {
                name: Some("Vec3".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: float_scalar,
                },
            },
            naga::Span::UNDEFINED,
        );

        let vec3i_ty = types.insert(
            Type {
                name: Some("Pos3".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::I32,
                },
            },
            naga::Span::UNDEFINED,
        );

        let vec3u_ty = types.insert(
            Type {
                name: Some("Vec3u".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar {
                        kind: ScalarKind::Uint,
                        width: 4,
                    },
                },
            },
            naga::Span::UNDEFINED,
        );

        let u32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar {
                    kind: ScalarKind::Uint,
                    width: 4,
                }),
            },
            naga::Span::UNDEFINED,
        );

        // Raw permutation data: arrays of 256 i32 values.
        let perm_array_ty = types.insert(
            Type {
                name: Some("PermutationTableData".into()),
                inner: TypeInner::Array {
                    base: i32_ty,
                    size: naga::ArraySize::Constant(core::num::NonZeroU32::new(256).unwrap()),
                    stride: 4, // i32 is 4 bytes
                },
            },
            naga::Span::UNDEFINED,
        );
        drop(types);

        // Perlin generator data that bundles permutation table and octave origins.
        let perm_table_ty = extern_converter.embed_wgsl_struct(&mut module, "PerlinNoiseGenerator");

        let types = &mut module.types;
        let output_ty = types.insert(
            Type {
                name: Some("Output".into()),
                inner: TypeInner::Array {
                    base: float_ty,
                    size: naga::ArraySize::Dynamic,
                    stride: 4,
                },
            },
            naga::Span::UNDEFINED,
        );

        TypeCache {
            float_ty,
            i32_ty,
            u32_ty,
            i64_ty,
            vec3f_ty,
            vec3i_ty,
            vec3u_ty,
            perm_array_ty,
            perm_table_ty,
            output_ty,
        }
    }

    /// Convert an SPMT variable type to the corresponding cached Naga type handle.
    pub fn convert_type(&self, t: &spmt::VariableType) -> Handle<Type> {
        match t {
            spmt::VariableType::DensityInput => self.float_ty,
            spmt::VariableType::F32 => self.float_ty,
            spmt::VariableType::Vec3 => self.vec3f_ty,
            spmt::VariableType::Pos3 => self.vec3u_ty,
            spmt::VariableType::I32 => self.u32_ty,
            spmt::VariableType::I64 => self.i64_ty,
            spmt::VariableType::PermutationTable => self.perm_table_ty,
        }
    }

    /// Create a density input array type (array<f32/f64, N>) and register it.
    pub fn make_density_array_type(
        &self,
        types: &mut naga::UniqueArena<Type>,
        length: u32,
    ) -> Handle<Type> {
        types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: self.float_ty,
                    size: naga::ArraySize::Constant(
                        core::num::NonZeroU32::new(length).expect("array length must be > 0"),
                    ),
                    stride: match types[self.float_ty].inner {
                        TypeInner::Scalar(s) => s.width as u32,
                        _ => unreachable!(),
                    },
                },
            },
            naga::Span::UNDEFINED,
        )
    }
}

/// Convert an SPMT binary operator to a Naga binary operator.
pub fn convert_binary_op(op: spmt::BinaryOperator) -> naga::BinaryOperator {
    match op {
        spmt::BinaryOperator::Add => naga::BinaryOperator::Add,
        spmt::BinaryOperator::Subtract => naga::BinaryOperator::Subtract,
        spmt::BinaryOperator::Multiply => naga::BinaryOperator::Multiply,
        spmt::BinaryOperator::Divide => naga::BinaryOperator::Divide,
        spmt::BinaryOperator::Equal => naga::BinaryOperator::Equal,
        spmt::BinaryOperator::NotEqual => naga::BinaryOperator::NotEqual,
        spmt::BinaryOperator::Less => naga::BinaryOperator::Less,
        spmt::BinaryOperator::LessEqual => naga::BinaryOperator::LessEqual,
        spmt::BinaryOperator::Greater => naga::BinaryOperator::Greater,
        spmt::BinaryOperator::GreaterEqual => naga::BinaryOperator::GreaterEqual,
        spmt::BinaryOperator::And => naga::BinaryOperator::LogicalAnd,
        spmt::BinaryOperator::Or => naga::BinaryOperator::LogicalOr,
    }
}

/// Convert an SPMT unary operator to a Naga unary operator.
pub fn convert_unary_op(op: spmt::UnaryOperator) -> naga::UnaryOperator {
    match op {
        spmt::UnaryOperator::Negate => naga::UnaryOperator::Negate,
    }
}

/// Generate a sanitized permutation table variable name, matching the RCL convention.
pub fn permutation_table_var_name(perm_table: &spmt::PermutationTableInput) -> String {
    sanitize_name(&format!(
        "perm_table_{}_{}_{}",
        perm_table.ident,
        perm_table.subident_index,
        perm_table.subident.as_ref().unwrap_or(&"".to_string())
    ))
}

/// Sanitize identifier names for use in shaders (replace non-alphanumeric chars with underscores).
pub fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
