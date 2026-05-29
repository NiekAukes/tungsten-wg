/*
Expression conversion from SPMT to CUDA C++.
*/

use std::rc::Rc;

use super::{CudaFunctionConverter, InputKey, function};
use crate::cuda::model as cuda;
use crate::spmt::model::{self as spmt, Addr};
use crate::transform_cuda::types::{self as types, convert_type, permutation_table_param_name};

impl<'a, 'm> CudaFunctionConverter<'m> {
    pub fn convert_expression(&mut self, expr: &spmt::Expression<'a>) -> cuda::Expression<'m> {
        match expr {
            spmt::Expression::Variable(var) => {
                let cuda_var = self.get_or_create_variable(var.clone());
                cuda::Expression::Variable(cuda_var)
            }

            spmt::Expression::Float(val) => cuda::Expression::F64Literal(*val as f64),
            spmt::Expression::Int(val) => cuda::Expression::I32Literal(*val),
            spmt::Expression::Long(val) => cuda::Expression::I64Literal(*val),

            spmt::Expression::BinaryOp { op, left, right } => {
                let left = Box::new(self.convert_expression(left));
                let right = Box::new(self.convert_expression(right));
                let op = types::convert_binary_op(*op);
                cuda::Expression::BinaryOp { op, left, right }
            }

            spmt::Expression::UnaryOp { op, operand } => {
                let operand = Box::new(self.convert_expression(operand));
                let op = types::convert_unary_op(*op);
                cuda::Expression::UnaryOp { op, operand }
            }

            spmt::Expression::Field { base, field, .. } => {
                let base = Box::new(self.convert_expression(base));
                cuda::Expression::Field {
                    base,
                    field: field.clone(),
                }
            }

            spmt::Expression::FunctionCall {
                function,
                parameters,
            } => {
                // Memoised: avoid reconverting the same helper function.
                let function_ref =
                    if let Some(func) = self.already_converted_functions.get(&function.addr()) {
                        func.clone()
                    } else {
                        let (cuda_func, converter) = function::convert_function(
                            function,
                            self.arena,
                            self.already_converted_functions.clone(),
                            self.density_function_inputs.clone(),
                        );
                        self.already_converted_functions
                            .insert(function.addr(), cuda_func.clone());
                        self.already_converted_functions
                            .extend(converter.already_converted_functions.into_iter());
                        cuda_func
                    };

                let mut arguments = vec![];
                for param in parameters {
                    arguments.push(self.convert_expression(param));
                }
                // Pass through any density-input variables as extra arguments.
                for (key, _) in self.density_function_inputs.as_ref() {
                    let v = self
                        .get_variable(key)
                        .expect("Density function input variable not found in converter");
                    arguments.push(cuda::Expression::Variable(v));
                }
                cuda::Expression::FunctionCall {
                    function: function_ref,
                    arguments,
                }
            }

            spmt::Expression::ExternCall {
                function_name,
                parameters,
                parameter_types,
            } => {
                // ExternCalls map to CUDA device math intrinsics / other helpers.
                // The C99 / CUDA math function names are intentionally the same
                // (fabs, floor, fma, sqrt, sin, cos, …) so no renaming is needed.
                let arguments = parameters
                    .iter()
                    .zip(parameter_types.iter())
                    .map(|(p, pt)| self.convert_argument(p, pt.clone()))
                    .collect();
                cuda::Expression::LateBoundCall {
                    function_name: function_name.clone(),
                    arguments,
                }
            }

            spmt::Expression::DensityVariable(input, index) => {
                // The density input is a flat device pointer `const double* input_N`.
                // We index it with a 1-D position derived from the thread's pos3.
                let flat_index = if let Some(idx_expr) = index {
                    self.convert_expression(idx_expr)
                } else {
                    // Compute flat index from the int3 pos3 parameter.
                    pos3_to_flat_index_expr(input.dimensions)
                };

                let param = self
                    .density_function_inputs
                    .get(&InputKey::from(input))
                    .cloned()
                    .unwrap_or_else(|| cuda::Parameter {
                        name: format!("err_{}", input.density_function.addr() as usize),
                        t: cuda::Type::ConstPointer(Box::new(cuda::Type::Float)),
                        is_const: true,
                    });

                cuda::Expression::Index {
                    base: Box::new(cuda::Expression::Variable(Rc::new(cuda::Variable {
                        name: spmt::Name::Named(param.name),
                        t: param.t,
                        memory_qualifier: None,
                    }))),
                    index: Box::new(flat_index),
                }
            }

            spmt::Expression::PermutationTable(input) => {
                // Passed as `const int8_t* perm_table_X` — return a variable reference.
                cuda::Expression::Variable(Rc::new(cuda::Variable {
                    name: spmt::Name::Named(permutation_table_param_name(input)),
                    t: cuda::Type::ConstPointer(Box::new(cuda::Type::Int8)),
                    memory_qualifier: None,
                }))
            }

            spmt::Expression::Construct { t, args } => {
                let converted_args = args.iter().map(|a| self.convert_expression(a)).collect();
                // Map to CUDA built-in constructor functions.
                let func_name = match t {
                    spmt::VariableType::Vec3 => "make_float3",
                    spmt::VariableType::Pos3 => "make_int3",
                    _ => panic!("Cannot construct {:?} with Construct expression", t),
                };
                cuda::Expression::LateBoundCall {
                    function_name: func_name.to_string(),
                    arguments: converted_args,
                }
            }

            spmt::Expression::ArrayAccess { array, index } => {
                let base = Box::new(self.convert_expression(array));
                let index = Box::new(self.convert_expression(index));
                cuda::Expression::Index { base, index }
            }
            spmt::Expression::ConstructExtern { t, args } => {
                let struct_name = match t {
                    spmt::VariableType::Extern(name) => name.to_string(),
                    other => panic!("ConstructExtern used with non-Extern type {:?}", other),
                };
                let fields = args
                    .iter()
                    .map(|(field, expr)| (field.to_string(), self.convert_expression(expr)))
                    .collect();
                cuda::Expression::StructInit {
                    struct_name,
                    fields,
                }
            }
            spmt::Expression::ArrayLiteral(exprs) => {
                let converted = exprs.iter().map(|e| self.convert_expression(e)).collect();
                cuda::Expression::ArrayLiteral(converted)
            }
        }
    }

    /// Convert an expression, casting its result to `target_type` when necessary.
    pub(crate) fn convert_argument(
        &mut self,
        arg: &spmt::Expression<'a>,
        target_type: spmt::VariableType,
    ) -> cuda::Expression<'m> {
        let expr = self.convert_expression(arg);
        let expected = convert_type(&target_type);
        // Only insert an explicit cast when the types differ.
        match &expr {
            cuda::Expression::F64Literal(_) | cuda::Expression::F32Literal(_) => {
                if expected == cuda::Type::Float {
                    return expr;
                }
            }
            cuda::Expression::I32Literal(_) => {
                if expected == cuda::Type::Int32 {
                    return expr;
                }
            }
            _ => {}
        }
        // Prefer a no-op identity cast rather than a semantic cast for the
        // common case where types already match; the CUDA compiler will fold it.
        expr
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a CUDA expression that computes the flat 1-D index into a density
/// array of the given `dimensions` from the `pos3` int3 parameter.
///
/// With dimensions (dx, dy, dz):
///   flat = pos3.x + pos3.y * dx + pos3.z * dx * dy
fn pos3_to_flat_index_expr<'m>(dimensions: (i32, i32, i32)) -> cuda::Expression<'m> {
    let (dx, dy, _dz) = dimensions;

    // pos3.x
    let x = cuda::Expression::Field {
        base: Box::new(pos3_var()),
        field: "x".to_string(),
    };
    // pos3.y * dx
    let y_row = cuda::Expression::BinaryOp {
        op: cuda::BinaryOperator::Multiply,
        left: Box::new(cuda::Expression::Field {
            base: Box::new(pos3_var()),
            field: "y".to_string(),
        }),
        right: Box::new(cuda::Expression::I32Literal(dx)),
    };
    // pos3.z * dx * dy
    let z_slice = cuda::Expression::BinaryOp {
        op: cuda::BinaryOperator::Multiply,
        left: Box::new(cuda::Expression::Field {
            base: Box::new(pos3_var()),
            field: "z".to_string(),
        }),
        right: Box::new(cuda::Expression::I32Literal(dx * dy)),
    };

    // x + (y * dx) + (z * dx * dy)
    cuda::Expression::BinaryOp {
        op: cuda::BinaryOperator::Add,
        left: Box::new(cuda::Expression::BinaryOp {
            op: cuda::BinaryOperator::Add,
            left: Box::new(x),
            right: Box::new(y_row),
        }),
        right: Box::new(z_slice),
    }
}

fn pos3_var<'m>() -> cuda::Expression<'m> {
    cuda::Expression::Variable(Rc::new(cuda::Variable {
        name: spmt::Name::Named("pos3".to_string()),
        t: cuda::Type::Struct("int3".to_string()),
        memory_qualifier: None,
    }))
}
