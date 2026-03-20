/*
Expression conversion from SPMT to RCL.
Handles conversion of all expression types including literals,
variables, binary/unary operations, function calls, and field access.
*/

use std::rc::Rc;

use super::{RCLFunctionConverter, sanitize_name, types};
use crate::rcl::{Parameter, model as rcl};
use crate::spmt::model::{self as spmt, Addr, Interned};
use crate::transform_rcl::function;
use crate::transform_rcl::types::permutation_table_var_name;

/// Convert an SPMT expression to an RCL expression

impl<'a, 'm> RCLFunctionConverter<'m> {
    pub fn convert_expression(&mut self, expr: &spmt::Expression<'a>) -> rcl::Expression<'m> {
        match expr {
            spmt::Expression::Variable(var) => {
                let rcl_var = self.get_or_create_variable(var.clone());

                rcl::Expression::Variable(rcl_var)
            }
            spmt::Expression::Float(val) => rcl::Expression::FloatLiteral(*val),
            spmt::Expression::Int(val) => rcl::Expression::I32Literal(*val),
            spmt::Expression::Long(val) => rcl::Expression::I64Literal(*val),
            spmt::Expression::BinaryOp { op, left, right } => {
                let left = Box::new(self.convert_expression(left));
                let right = Box::new(self.convert_expression(right));
                let op = types::convert_binary_op(*op);
                rcl::Expression::BinaryOp { op, left, right }
            }
            spmt::Expression::UnaryOp { op, operand } => {
                let operand = Box::new(self.convert_expression(operand));
                let op = types::convert_unary_op(*op);
                rcl::Expression::UnaryOp { op, operand }
            }
            spmt::Expression::Field { base, field } => {
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
                        let (rcl_func, converter) = function::convert_function(
                            function,
                            self.arena,
                            self.already_converted_functions.clone(),
                            self.density_function_inputs.clone(),
                        );
                        self.already_converted_functions
                            .insert(function.addr(), rcl_func.clone());
                        self.already_converted_functions
                            .extend(converter.already_converted_functions.into_iter());
                        rcl_func
                    };

                let mut arguments = vec![];
                for param in parameters {
                    arguments.push(self.convert_expression(param));
                }
                for (key, _) in self.density_function_inputs.as_ref() {
                    let v = self
                        .get_variable(*key)
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
                    arguments.push(self.convert_argument(param, parameter_types[i]));
                    argument_types.push(types::convert_type(&parameter_types[i]));
                }

                rcl::Expression::LateBoundCall {
                    function_name: function_name.clone(),
                    argument_types: argument_types,
                    return_type: rcl::Type::F32,
                    arguments,
                }
            }
            spmt::Expression::DensityVariable(input) => {
                let onedposition = convert_vec3_to_position_expression(
                    rcl::Expression::Variable(Rc::new(rcl::Variable {
                        name: Some("pos3".to_string()),
                        t: rcl::Type::Struct("Pos3".to_string()),
                        mutable: false,
                    })),
                    input.dimensions,
                );

                // the function is a density function, which is passed as a
                // field of f32s, look up the parameter for this density function in the converter's density_function_inputs map
                let function_param = self
                    .density_function_inputs
                    .get(&input.density_function.addr())
                    //.expect("Density function input not found in converter")
                    .cloned()
                    .unwrap_or(Parameter {
                        name: format!("err_{}", input.density_function.addr() as usize),
                        t: rcl::Type::ArrayRef(Box::new(rcl::Type::F32), 16 * 16 * 256),
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
                    spmt::VariableType::I32 => panic!(
                        "Cannot construct I32 directly, it should be a literal or a variable"
                    ),
                    spmt::VariableType::I64 => panic!(
                        "Cannot construct I64 directly, it should be a literal or a variable"
                    ),
                    spmt::VariableType::DensityInput => "DensityInput",
                    spmt::VariableType::PermutationTable => "PermutationTable",
                };
                rcl::Expression::LateBoundCall {
                    function_name: format!("{}::new", type_name),
                    argument_types: vec![types::convert_type(t); args.len()],
                    return_type: types::convert_type(t),
                    arguments: converted_args,
                }
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
                to_type: rcl::Type::F32,
            },
            (spmt::VariableType::I32, _) => rcl::Expression::Cast {
                expr: Box::new(exp),
                to_type: rcl::Type::I32,
            },
            (_, _) => exp, // do nothing
        };
        cast
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
