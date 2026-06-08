/*
Expression conversion from SPMT to RCL.
Handles conversion of all expression types including literals,
variables, binary/unary operations, function calls, and field access.
*/

use std::rc::Rc;

use super::{RCLFunctionConverter, sanitize_name, types};
use crate::rcl::{Parameter, model as rcl};
use crate::spmt::model::{self as spmt, Addr, Interned};
use crate::spmt::try_derive_type;
use crate::transform_rcl::types::{convert_type, permutation_table_var_name};
use crate::transform_rcl::{InputKey, function};

/// Convert an SPMT expression to an RCL expression

impl<'a, 'm> RCLFunctionConverter<'m> {
    pub fn convert_expression(&mut self, expr: &spmt::Expression<'a>) -> rcl::Expression<'m> {
        match expr {
            spmt::Expression::Variable(var) => {
                let rcl_var = self.get_or_create_variable(var.clone());

                rcl::Expression::Variable(rcl_var)
            }
            spmt::Expression::Float(val) => rcl::Expression::F32Literal(*val),
            spmt::Expression::Double(val) => rcl::Expression::F64Literal(*val),
            spmt::Expression::Int(val) => rcl::Expression::I32Literal(*val),
            spmt::Expression::Long(val) => rcl::Expression::I64Literal(*val),
            spmt::Expression::BinaryOp { op, left, right } => {
                // let left = Box::new(self.convert_expression(left));
                // let right = Box::new(self.convert_expression(right));
                let (left, right) = self.try_convert_arguments_binary_op(left, right, op);
                let op = types::convert_binary_op(*op);
                rcl::Expression::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            spmt::Expression::UnaryOp { op, operand } => {
                let operand = Box::new(self.convert_expression(operand));
                let op = types::convert_unary_op(*op);
                rcl::Expression::UnaryOp { op, operand }
            }
            spmt::Expression::Field {
                base,
                field,
                type_of_field: _,
                known_idnex: _,
            } => {
                let base = Box::new(self.convert_expression(base));
                rcl::Expression::Field {
                    base,
                    field: field.clone(),
                }
            }
            spmt::Expression::FunctionCall {
                function,
                parameters,
            } => {
                // check if the function is already converted
                let function_ref =
                    if let Some(func) = self.already_converted_functions.get(&function.addr()) {
                        func.clone()
                    } else {
                        // not converted yet, convert the function and store it in the map
                        let (rcl_func, converter, is_new) = function::convert_function(
                            function,
                            self.arena,
                            self.already_converted_functions.clone(),
                            self.density_function_inputs.clone(),
                            self.density_func_name.clone(),
                        );
                        if is_new {
                            self.already_converted_functions
                                .insert(function.addr(), rcl_func.clone());
                            self.already_converted_functions
                                .extend(converter.already_converted_functions.into_iter());
                        }
                        rcl_func
                    };

                let mut arguments = vec![];
                for param in parameters {
                    arguments.push(self.convert_expression(param));
                }
                for (key, _) in self.density_function_inputs.as_ref() {
                    let v = self
                        .get_variable(key)
                        .expect("Density function input variable not found in converter");
                    arguments.push(rcl::Expression::Variable(v));
                }
                rcl::Expression::FunctionCall {
                    function: function_ref,
                    arguments,
                }
            }
            spmt::Expression::ExternCall {
                function_name,
                parameters,
                parameter_types,
            } => {
                // not a real extern call, it is simply a function call to a function that is not defined in the SPMT,
                // so we treat it as a "late bound" function call with the given name and parameters

                // let arguments = parameters
                //     .iter()
                //     .map(|p| self.convert_expression(p))
                //     .collect();
                let mut arguments = vec![];
                let mut argument_types = vec![];
                for (i, param) in parameters.iter().enumerate() {
                    arguments.push(self.convert_argument(param, parameter_types[i].clone()));
                    argument_types.push(types::convert_type(&parameter_types[i]));
                }

                rcl::Expression::LateBoundCall {
                    function_name: function_name.clone(),
                    argument_types: argument_types,
                    return_type: convert_type(&spmt::VariableType::F32),
                    arguments,
                }
            }
            spmt::Expression::DensityVariable(input, index) => {
                let onedposition = if index.is_some() {
                    self.convert_expression(index.as_ref().unwrap())
                } else {
                    convert_vec3_to_position_expression(
                        rcl::Expression::Variable(Rc::new(rcl::Variable {
                            name: Some("pos3".to_string()),
                            t: rcl::Type::Struct("Pos3".to_string()),
                            mutable: false,
                        })),
                        input.dimensions,
                    )
                };

                // the function is a density function, which is passed as a
                // field of f32s, look up the parameter for this density function in the converter's density_function_inputs map
                let function_param = self
                    .density_function_inputs
                    .get(&InputKey::from(input))
                    //.expect("Density function input not found in converter")
                    .cloned()
                    .unwrap_or(Parameter {
                        name: format!("err_{}", input.density_function.addr() as usize),
                        t: rcl::Type::Void,
                    });
                rcl::Expression::Index {
                    base: Box::new(rcl::Expression::Variable(Rc::new(rcl::Variable {
                        name: Some(function_param.name.clone()),
                        t: function_param.t.clone(),
                        mutable: false,
                    }))),
                    index: Box::new(onedposition),
                }
            }
            spmt::Expression::PermutationTable(input) => {
                // similar to density variable, but we don't need to convert the position to a 1D index, we just pass the position as a Vec3
                rcl::Expression::Variable(Rc::new(rcl::Variable {
                    name: Some(permutation_table_var_name(input)),
                    t: rcl::Type::Array(Box::new(rcl::Type::I8), 256), // permutation tables are passed as arrays of 256 i8s
                    mutable: false,
                }))
            }

            // spmt::Expression::MakeVec3 { x, y, z } => {
            //     // make a function call to Vec3::new(x,y,z)
            //     let x = self.convert_expression(x);
            //     let y = self.convert_expression(y);
            //     let z = self.convert_expression(z);
            //     rcl::Expression::LateBoundCall {
            //         function_name: "Vec3::new".to_string(),
            //         argument_types: vec![rcl::Type::F32, rcl::Type::F32, rcl::Type::F32],
            //         return_type: rcl::Type::Vec3,
            //         arguments: vec![x, y, z],
            //     }
            // }
            spmt::Expression::Construct { t, args } => {
                let converted_args = args
                    .iter()
                    .map(|arg| self.convert_expression(arg))
                    .collect();
                let type_name = match t {
                    spmt::VariableType::Vec3 => "Vec3",
                    spmt::VariableType::Pos3 => "Pos3",
                    spmt::VariableType::F32 => panic!(
                        "Cannot construct F32 directly, it should be a literal or a variable"
                    ),
                    spmt::VariableType::F64 => panic!(
                        "Cannot construct F64 directly, it should be a literal or a variable"
                    ),
                    spmt::VariableType::I32 => panic!(
                        "Cannot construct I32 directly, it should be a literal or a variable"
                    ),
                    spmt::VariableType::I64 => panic!(
                        "Cannot construct I64 directly, it should be a literal or a variable"
                    ),
                    spmt::VariableType::DensityInput => "DensityInput",
                    spmt::VariableType::PermutationTable => "PermutationTable",
                    spmt::VariableType::Extern(s) => s,
                    spmt::VariableType::Array(element_type, size) => {
                        panic!(
                            "Cannot construct array types directly, they should be constructed using array literals or other means"
                        )
                    }
                    spmt::VariableType::Bool => panic!(
                        "Cannot construct Bool directly, it should be a literal or a variable"
                    ),
                };
                rcl::Expression::LateBoundCall {
                    function_name: format!("{}::new", type_name),
                    argument_types: vec![types::convert_type(t); args.len()],
                    return_type: types::convert_type(t),
                    arguments: converted_args,
                }
            }
            spmt::Expression::ArrayAccess { array, index } => {
                let array = Box::new(self.convert_expression(array));
                let index = Box::new(self.convert_expression(index));
                rcl::Expression::Index { base: array, index }
            }
            spmt::Expression::ConstructExtern { t, args } => {
                let converted_args = args
                    .iter()
                    .map(|(name, arg)| (*name, self.convert_expression(arg)))
                    .collect();
                rcl::Expression::Construct {
                    t: convert_type(t),
                    args: converted_args,
                }
            }
            spmt::Expression::ArrayLiteral(expressions) => {
                let converted_elements: Vec<rcl::Expression<'_>> = expressions
                    .iter()
                    .map(|expr| self.convert_expression(expr))
                    .collect();
                if expressions.len() != converted_elements.len() {
                    panic!("Array literal conversion error: length mismatch");
                }
                rcl::Expression::ArrayLiteral(converted_elements)
            }
        }
    }

    fn convert_argument(
        &mut self,
        arg: &spmt::Expression<'a>,
        target_type: spmt::VariableType,
    ) -> rcl::Expression<'m> {
        let exp = self.convert_expression(arg);
        // if the target type is f32 and the expression is an int literal, convert it to a float literal
        let cast = match (target_type, arg) {
            (spmt::VariableType::F32, spmt::Expression::Float(_)) => exp,
            (spmt::VariableType::I32, spmt::Expression::Int(_)) => exp,
            (spmt::VariableType::F32, _) => rcl::Expression::Cast {
                expr: Box::new(exp),
                to_type: convert_type(&spmt::VariableType::F32),
            },
            (spmt::VariableType::I32, _) => rcl::Expression::Cast {
                expr: Box::new(exp),
                to_type: convert_type(&spmt::VariableType::I32),
            },
            (_, _) => exp, // do nothing
        };
        cast
    }

    pub fn try_convert_arguments_binary_op(
        &mut self,
        left: &spmt::Expression<'a>,
        right: &spmt::Expression<'a>,
        op: &spmt::BinaryOperator,
    ) -> (rcl::Expression<'m>, rcl::Expression<'m>) {
        // try to derive the types of the left and right expressions
        let l = try_derive_type(&left);
        let r = try_derive_type(&right);

        let (left_h, right_h) = (
            self.convert_expression(&left),
            self.convert_expression(&right),
        );

        let (ltype, rtype) = match (l, r) {
            (Some(lt), Some(rt)) => (lt, rt),
            _ => {
                return (
                    self.convert_expression(&left),
                    self.convert_expression(&right),
                );
            } // if we can't derive types, just convert without trying to match
        };

        match (op, ltype, rtype) {
            // If one side is vec3<f32> and the other is vec3<i32>, convert the i32 to f32
            (
                spmt::BinaryOperator::Add
                | spmt::BinaryOperator::Subtract
                | spmt::BinaryOperator::Multiply
                | spmt::BinaryOperator::Divide,
                spmt::VariableType::I32,
                spmt::VariableType::Bool,
            ) => (
                left_h,
                rcl::Expression::Cast {
                    expr: Box::new(right_h),
                    to_type: convert_type(&spmt::VariableType::I32),
                },
            ),
            _ => (left_h, right_h), // No conversion needed or possible
        }
    }
    pub fn convert_expression_to_type(
        &mut self,
        source: &spmt::Expression<'a>,
        target_type: spmt::VariableType,
    ) -> rcl::Expression<'m> {
        let source_h = self.convert_expression(source);
        let source_type = match try_derive_type(source) {
            Some(t) => t,
            None => return source_h, // If we can't derive the type, just convert without trying to match
        };
        if source_type == target_type {
            source_h
        } else {
            rcl::Expression::Cast {
                expr: Box::new(source_h),
                to_type: convert_type(&target_type),
            }
        }
    }
}

fn convert_vec3_to_position_expression<'m>(
    p: rcl::Expression<'m>,
    dims: (i32, i32, i32),
) -> rcl::Expression<'m> {
    // // simply y * 256 * 16 + z * 256 + x
    // rcl::Expression::BinaryOp {
    //     op: rcl::BinaryOperator::Modulo,
    //     left: Box::new(rcl::Expression::BinaryOp {
    //         op: rcl::BinaryOperator::Add,
    //         left: Box::new(rcl::Expression::BinaryOp {
    //             op: rcl::BinaryOperator::Add,
    //             left: Box::new(rcl::Expression::BinaryOp {
    //                 op: rcl::BinaryOperator::Multiply,
    //                 left: Box::new(rcl::Expression::Field {
    //                     base: Box::new(p.clone()),
    //                     field: "y".to_string(),
    //                 }),
    //                 right: Box::new(rcl::Expression::IntLiteral(256 * 16)),
    //             }),
    //             right: Box::new(rcl::Expression::BinaryOp {
    //                 op: rcl::BinaryOperator::Multiply,
    //                 left: Box::new(rcl::Expression::Field {
    //                     base: Box::new(p.clone()),
    //                     field: "z".to_string(),
    //                 }),
    //                 right: Box::new(rcl::Expression::IntLiteral(256)),
    //             }),
    //         }),
    //         right: Box::new(rcl::Expression::Field {
    //             base: Box::new(p),
    //             field: "x".to_string(),
    //         }),
    //     }),
    //     right: Box::new(rcl::Expression::IntLiteral(16 * 256 * 256)),
    // }

    // new method: as_index(pos3, x, y)
    rcl::Expression::LateBoundCall {
        function_name: "as_index".to_string(),
        argument_types: vec![
            rcl::Type::Struct("Pos3".to_string()),
            rcl::Type::I32,
            rcl::Type::I32,
        ],
        return_type: rcl::Type::F32,
        arguments: vec![
            p,
            rcl::Expression::I32Literal(dims.0),
            rcl::Expression::I32Literal(dims.1),
        ],
    }
}
