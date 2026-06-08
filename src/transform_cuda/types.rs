/*
Type conversion utilities for translating SPMT types to CUDA C++ types.
*/

use crate::cuda::model as cuda;
use crate::spmt::model as spmt;

/// Convert an SPMT variable type to a CUDA type.
pub fn convert_type(t: &spmt::VariableType) -> cuda::Type {
    match t {
        // Density values use double precision on the device.
        spmt::VariableType::DensityInput => cuda::Type::Float,
        spmt::VariableType::F32 => cuda::Type::Float,
        spmt::VariableType::F64 => cuda::Type::Double,
        spmt::VariableType::I32 => cuda::Type::Int32,
        spmt::VariableType::I64 => cuda::Type::Int64,
        // CUDA built-in vector types.
        spmt::VariableType::Vec3 => cuda::Type::Struct("float3".to_string()),
        spmt::VariableType::Pos3 => cuda::Type::Struct("int3".to_string()),
        // Permutation tables are passed as a raw const pointer.
        spmt::VariableType::PermutationTable => {
            cuda::Type::ConstPointer(Box::new(cuda::Type::Int8))
        }
        spmt::VariableType::Bool => cuda::Type::Int32,
        spmt::VariableType::Array(elem_type, size) => {
            let elem_type = Box::new(convert_type(elem_type));
            cuda::Type::Array(elem_type, *size)
        }
        spmt::VariableType::Extern(name) => cuda::Type::Struct(name.to_string()),
    }
}

/// Convert an SPMT binary operator to a CUDA binary operator.
pub fn convert_binary_op(op: spmt::BinaryOperator) -> cuda::BinaryOperator {
    match op {
        spmt::BinaryOperator::Add => cuda::BinaryOperator::Add,
        spmt::BinaryOperator::Subtract => cuda::BinaryOperator::Subtract,
        spmt::BinaryOperator::Multiply => cuda::BinaryOperator::Multiply,
        spmt::BinaryOperator::Divide => cuda::BinaryOperator::Divide,
        spmt::BinaryOperator::Equal => cuda::BinaryOperator::Equal,
        spmt::BinaryOperator::NotEqual => cuda::BinaryOperator::NotEqual,
        spmt::BinaryOperator::Less => cuda::BinaryOperator::Less,
        spmt::BinaryOperator::LessEqual => cuda::BinaryOperator::LessEqual,
        spmt::BinaryOperator::Greater => cuda::BinaryOperator::Greater,
        spmt::BinaryOperator::GreaterEqual => cuda::BinaryOperator::GreaterEqual,
        spmt::BinaryOperator::And => cuda::BinaryOperator::And,
        spmt::BinaryOperator::Or => cuda::BinaryOperator::Or,
    }
}

/// Convert an SPMT unary operator to a CUDA unary operator.
pub fn convert_unary_op(op: spmt::UnaryOperator) -> cuda::UnaryOperator {
    match op {
        spmt::UnaryOperator::Negate => cuda::UnaryOperator::Negate,
    }
}

/// Generate the CUDA parameter name for a permutation table input.
pub fn permutation_table_param_name(perm_table: &spmt::PermutationTableInput) -> String {
    crate::transform_cuda::sanitize_name(&format!(
        "perm_table_{}_{}_{}",
        perm_table.ident,
        perm_table.subident_index,
        perm_table
            .subident
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("")
    ))
}
