/*
Type conversion utilities for translating SPMT types to RCL types.
*/

use crate::rcl::model as rcl;
use crate::spmt::model as spmt;
use crate::transform_rcl::sanitize_name;

/// Convert an SPMT variable type to an RCL type
pub fn convert_type(t: &spmt::VariableType) -> rcl::Type {
    match t {
        spmt::VariableType::DensityInput => rcl::Type::F64,
        spmt::VariableType::Vec3 => rcl::Type::Struct("Vec3".to_string()),
        spmt::VariableType::Pos3 => rcl::Type::Struct("Pos3".to_string()),
        spmt::VariableType::F32 => rcl::Type::F64,
        spmt::VariableType::I32 => rcl::Type::I32,
        spmt::VariableType::I64 => rcl::Type::I64,
        spmt::VariableType::PermutationTable => rcl::Type::Ref(Box::new(rcl::Type::Struct(
            crate::transform_rcl::PERLIN_NOISE_SAMPLER_STRUCT_NAME.to_string(),
        ))),
        spmt::VariableType::Extern(name) => rcl::Type::Struct(sanitize_name(name)),
        spmt::VariableType::Array(element_type, size) => {
            // For simplicity, we can represent arrays as structs with fields like element_0, element_1, etc.
            rcl::Type::Array(Box::new(convert_type(element_type)), *size)
        }
        spmt::VariableType::Bool => rcl::Type::Bool,
    }
}

/// Convert an SPMT binary operator to an RCL binary operator
pub fn convert_binary_op(op: spmt::BinaryOperator) -> rcl::BinaryOperator {
    match op {
        spmt::BinaryOperator::Add => rcl::BinaryOperator::Add,
        spmt::BinaryOperator::Subtract => rcl::BinaryOperator::Subtract,
        spmt::BinaryOperator::Multiply => rcl::BinaryOperator::Multiply,
        spmt::BinaryOperator::Divide => rcl::BinaryOperator::Divide,
        spmt::BinaryOperator::Equal => rcl::BinaryOperator::Equal,
        spmt::BinaryOperator::NotEqual => rcl::BinaryOperator::NotEqual,
        spmt::BinaryOperator::Less => rcl::BinaryOperator::Less,
        spmt::BinaryOperator::LessEqual => rcl::BinaryOperator::LessEqual,
        spmt::BinaryOperator::Greater => rcl::BinaryOperator::Greater,
        spmt::BinaryOperator::GreaterEqual => rcl::BinaryOperator::GreaterEqual,
        spmt::BinaryOperator::And => rcl::BinaryOperator::And,
        spmt::BinaryOperator::Or => rcl::BinaryOperator::Or,
    }
}

/// Convert an SPMT unary operator to an RCL unary operator
pub fn convert_unary_op(op: spmt::UnaryOperator) -> rcl::UnaryOperator {
    match op {
        spmt::UnaryOperator::Negate => rcl::UnaryOperator::Negate,
    }
}

pub fn permutation_table_var_name(perm_table: &spmt::PermutationTableInput) -> String {
    sanitize_name(&format!(
        "perm_table_{}_{}_{}",
        perm_table.ident,
        perm_table.subident_index,
        perm_table.subident.as_ref().unwrap_or(&"".to_string())
    ))
}
