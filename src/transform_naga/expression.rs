/*
Expression conversion from SPMT to Naga IR.
Handles conversion of all expression types including literals,
variables, binary/unary operations, function calls, field access,
density variable indexing, and vector construction.

Naga requires expressions to be emitted (via Statement::Emit) before they
can be used. This module tracks expression ranges for proper emit generation.
*/

use core::{f32, f64};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use naga::{Expression, Function, Handle, Literal, MathFunction, Span, Statement, Type, TypeInner};

use super::types::{TypeCache, convert_binary_op, convert_unary_op, permutation_table_var_name};
use super::{InputKey, NagaFunctionConverter};
use crate::spmt::model::{self as spmt, Addr};
use crate::spmt::try_derive_type;

/// Context for converting expressions within a single function.
/// Bundles the function-local state needed during expression conversion.
pub struct ExprContext<'m, 'a, 'b> {
    pub func: Option<&'b mut Function>,
    pub module: Rc<RefCell<naga::Module>>,
    pub type_cache: &'m TypeCache,
    pub converter: &'b mut NagaFunctionConverter<'m>,
    pub already_converted: Vec<(spmt::Expression<'a>, Handle<Expression>)>,
    pub handle_cache: Vec<(spmt::Expression<'a>, Handle<Expression>)>,
}

impl<'m, 'a, 'b> ExprContext<'m, 'a, 'b> {
    pub fn new(
        func: &'b mut Function,
        module: Rc<RefCell<naga::Module>>,
        type_cache: &'m TypeCache,
        converter: &'b mut NagaFunctionConverter<'m>,
    ) -> Self {
        Self {
            func: Some(func),
            module,
            type_cache,
            converter,
            already_converted: Vec::new(),
            handle_cache: Vec::new(),
        }
    }

    pub fn new_constant_context(
        module: Rc<RefCell<naga::Module>>,
        type_cache: &'m TypeCache,
        converter: &'b mut NagaFunctionConverter<'m>,
    ) -> Self {
        Self {
            func: None,
            module,
            type_cache,
            converter,
            already_converted: Vec::new(),
            handle_cache: Vec::new(),
        }
    }

    /// Append an expression and immediately emit it.
    pub(crate) fn append_and_emit(&mut self, expr: Expression) -> Handle<Expression> {
        //let old_len = self.func.expressions.len();
        //let h = self.func.expressions.append(expr, Span::UNDEFINED);
        //let range = self.func.expressions.range_from(old_len);
        //self.func.body.push(Statement::Emit(range), Span::UNDEFINED);
        self.append(expr)
    }

    pub(crate) fn append(&mut self, expr: Expression) -> Handle<Expression> {
        //self.func.expressions.append(expr, Span::UNDEFINED)
        match &mut self.func {
            Some(func) => {
                let h = func.expressions.append(expr, Span::UNDEFINED);
                h
            }
            None => {
                let h = self
                    .module
                    .borrow_mut()
                    .global_expressions
                    .append(expr, Span::UNDEFINED);
                h
            }
        }
    }

    /// Convert an SPMT expression to a Naga expression handle.
    /// The expression is appended to the function's arena and emitted as needed.
    pub fn convert_expression(&mut self, expr: &spmt::Expression<'a>) -> Handle<Expression> {
        // Check if we've already converted this expression (e.g. common subexpression)
        if let Some(&h) = self
            .already_converted
            .iter()
            .find_map(|(e, h)| if e == expr { Some(h) } else { None })
        {
            return h;
        }
        let r = match expr {
            spmt::Expression::Variable(var) => {
                let key = InputKey::from(*var);
                if let Some(&handle) = self.converter.var_map.get(&key) {
                    if self.converter.value_vars.contains(&key) {
                        // Direct value (e.g. function argument) - no Load needed

                        handle
                    } else {
                        // Local variable pointer — load the value
                        self.append(Expression::Load { pointer: handle })
                    }
                } else {
                    let var_name = self.converter.get_concrete_name(*var);
                    println!("Variable '{}' not found in converter var_map", var_name);
                    println!("Current var_map keys: {:?}", self.converter.var_map.keys());
                    panic!("Variable '{}' not found in converter var_map", var_name);
                }
            }

            spmt::Expression::Float(val) => {
                let lit = match self.type_cache.float_ty {
                    ty if self.module.borrow().types[ty].inner
                        == naga::TypeInner::Scalar(naga::Scalar {
                            kind: naga::ScalarKind::Float,
                            width: 8,
                        }) =>
                    {
                        // WGSL doesn't support f64 literals, so just use the max/min finite values for infinity to avoid validation errors.
                        let fixed_val = if *val == f32::INFINITY {
                            f64::MAX
                        } else if *val == f32::NEG_INFINITY {
                            f64::MIN
                        } else {
                            *val as f64
                        };
                        Literal::F64(fixed_val)
                    }
                    _ => {
                        if *val == f32::INFINITY {
                            Literal::F32(f32::MAX)
                        } else if *val == f32::NEG_INFINITY {
                            Literal::F32(f32::MIN)
                        } else {
                            Literal::F32(*val as f32)
                        }
                    }
                };
                self.append_and_emit(Expression::Literal(lit))
            }

            spmt::Expression::Double(val) => {
                let lit = match self.type_cache.float_ty {
                    ty if self.module.borrow().types[ty].inner
                        == naga::TypeInner::Scalar(naga::Scalar {
                            kind: naga::ScalarKind::Float,
                            width: 8,
                        }) =>
                    {
                        // WGSL doesn't support f64 literals, so just use the max/min finite values for infinity to avoid validation errors.
                        let fixed_val = if *val == f64::INFINITY {
                            f64::MAX
                        } else if *val == f64::NEG_INFINITY {
                            f64::MIN
                        } else {
                            *val as f64
                        };
                        Literal::F64(fixed_val)
                    }
                    _ => {
                        if *val == f64::INFINITY {
                            Literal::F32(f32::MAX)
                        } else if *val == f64::NEG_INFINITY {
                            Literal::F32(f32::MIN)
                        } else {
                            Literal::F32(*val as f32)
                        }
                    }
                };
                self.append_and_emit(Expression::Literal(lit))
            }

            spmt::Expression::Int(val) => {
                self.append_and_emit(Expression::Literal(Literal::U32(*val as u32)))
            }

            spmt::Expression::Long(val) => {
                self.append_and_emit(Expression::Literal(Literal::I64(*val)))
            }

            spmt::Expression::BinaryOp { op, left, right } => {
                // let left_h = self.convert_expression(left);
                // let right_h = self.convert_expression(right);
                let (left_h, right_h) = self.try_convert_arguments_binary_op(left, right, op);

                let naga_op = convert_binary_op(*op);
                self.append_and_emit(Expression::Binary {
                    op: naga_op,
                    left: left_h,
                    right: right_h,
                })
            }

            spmt::Expression::UnaryOp { op, operand } => {
                let operand_h = self.convert_expression(operand);
                let naga_op = convert_unary_op(*op);
                self.append_and_emit(Expression::Unary {
                    op: naga_op,
                    expr: operand_h,
                })
            }

            spmt::Expression::Field {
                base,
                field,
                type_of_field: _,
                known_idnex,
            } => {
                let base_h = self.convert_expression(base);
                if let Some(index) = known_idnex {
                    // If we know the index of the field, we can directly access it without matching on the field name.
                    self.append_and_emit(Expression::AccessIndex {
                        base: base_h,
                        index: *index as u32,
                    })
                } else {
                    // Otherwise, we match on the field name to determine the index.
                    let index = match field.as_str() {
                        "x" => 0,
                        "y" => 1,
                        "z" => 2,
                        "w" => 3,
                        _ => panic!("Unknown field name: {}", field),
                    };
                    self.append_and_emit(Expression::AccessIndex {
                        base: base_h,
                        index,
                    })
                }
            }

            spmt::Expression::FunctionCall {
                function,
                parameters,
            } => {
                // Check if function already converted
                let func_handle = if let Some(&h) = self
                    .converter
                    .already_converted_functions
                    .get(&function.addr())
                {
                    h
                } else {
                    // Need to convert the helper function first.
                    // We do this by calling function::convert_function at the module level.
                    // This requires splitting the borrow, so we collect state needed.
                    let mut sub_converter = self.converter.derive_new_with_state();
                    let h = super::function::convert_function(
                        function,
                        self.module.clone(),
                        self.type_cache,
                        &mut sub_converter,
                    );
                    // Merge converted functions back
                    self.converter
                        .already_converted_functions
                        .extend(sub_converter.already_converted_functions.into_iter());
                    h
                };

                // Convert arguments
                let mut arguments = Vec::new();
                for param in parameters {
                    arguments.push(self.convert_expression(param));
                }

                // If this function also needs density inputs forwarded, forward them.
                // (Helper functions that need density inputs get them passed through.)
                // For now, we only forward explicit parameters.

                // Create CallResult expression
                let result_expr = self
                    .func
                    .as_mut()
                    .unwrap()
                    .expressions
                    .append(Expression::CallResult(func_handle), Span::UNDEFINED);

                // Emit the Call statement
                self.func.as_mut().unwrap().body.push(
                    Statement::Call {
                        function: func_handle,
                        arguments,
                        result: Some(result_expr),
                    },
                    Span::UNDEFINED,
                );

                result_expr
            }

            spmt::Expression::ExternCall {
                function_name,
                parameters,
                parameter_types,
            } => {
                // Try to map known extern calls to naga MathFunction builtins
                if let Some(math_fn) = try_map_math_function(function_name) {
                    return self.convert_math_call(math_fn, parameters, parameter_types);
                }

                // Otherwise, generate or look up an extern function declaration

                let func_handle = self
                    .converter
                    .extern_converter
                    .embed_wgsl_function(self.module.borrow_mut(), function_name);

                let mut arguments = Vec::new();
                for (param, param_type) in parameters.iter().zip(parameter_types.iter()) {
                    let converted = self.convert_expression(param);
                    // Cast if needed (e.g., f64 literals to f32 for GPU)
                    arguments.push(self.maybe_cast(converted, param_type));
                }

                let result_expr = self
                    .func
                    .as_mut()
                    .unwrap()
                    .expressions
                    .append(Expression::CallResult(func_handle), Span::UNDEFINED);
                self.func.as_mut().unwrap().body.push(
                    Statement::Call {
                        function: func_handle,
                        arguments,
                        result: Some(result_expr),
                    },
                    Span::UNDEFINED,
                );

                result_expr
            }

            spmt::Expression::DensityVariable(input, index) => {
                let key = InputKey::from(input);
                let arg_info = self
                    .converter
                    .density_arg_map
                    .get(&key)
                    .unwrap_or_else(|| {
                        panic!(
                            "Density input argument for key {:?} not found in converter \n Density args: {:?}",
                            key,
                            self.converter.density_arg_map.keys()
                        )
                    });

                // Get the function argument expression (pointer to array)
                let g_var_expr = self.func.as_mut().unwrap().expressions.append(
                    Expression::GlobalVariable(arg_info.variable),
                    Span::UNDEFINED,
                );
                let member_ptr = self.append_and_emit(Expression::AccessIndex {
                    base: g_var_expr,
                    index: arg_info.member_index,
                });
                // Compute the index expression
                let index_expr = if let Some(idx) = index {
                    self.convert_expression(idx)
                } else {
                    // Default: compute 1D index from pos3 and dimensions
                    let dim_y = self.append_and_emit(Expression::Literal(Literal::U32(
                        input.dimensions.1 as u32,
                    )));
                    let dim_z = self.append_and_emit(Expression::Literal(Literal::U32(
                        input.dimensions.2 as u32,
                    )));
                    self.convert_default_density_index(dim_y, dim_z)
                };

                // Array access
                let elem_ptr = self.append_and_emit(Expression::Access {
                    base: member_ptr,
                    index: index_expr,
                });
                self.append_and_emit(Expression::Load { pointer: elem_ptr })
            }

            spmt::Expression::PermutationTable(input) => {
                let name = permutation_table_var_name(input);
                let arg_info = self
                    .converter
                    .perm_table_arg_map
                    .get(&name)
                    .unwrap_or_else(|| {
                        panic!(
                            "Permutation table argument '{}' not found in converter \n Permutation tables: {:?}",
                            name,
                            self.converter.perm_table_arg_map
                        )
                    });

                let g_var_expr = self.func.as_mut().unwrap().expressions.append(
                    Expression::GlobalVariable(arg_info.variable),
                    Span::UNDEFINED,
                );
                let member_ptr = self.append_and_emit(Expression::AccessIndex {
                    base: g_var_expr,
                    index: arg_info.member_index,
                });
                self.append_and_emit(Expression::Load {
                    pointer: member_ptr,
                })
            }

            spmt::Expression::Construct { t, args } => {
                let naga_ty = self.type_cache.convert_type_full(
                    &mut self.module.borrow_mut(),
                    self.converter.extern_converter,
                    t,
                );
                let mut components = Vec::new();
                for arg in args {
                    components.push(self.convert_expression(arg));
                }
                self.append_and_emit(Expression::Compose {
                    ty: naga_ty,
                    components,
                })
            }
            spmt::Expression::ArrayAccess { array, index } => {
                let array_h = self.convert_expression(array);
                let index_h = self.convert_expression(index);
                let element_ptr = self.append_and_emit(Expression::Access {
                    base: array_h,
                    index: index_h,
                });
                // self.append_and_emit(Expression::Load {
                //     pointer: element_ptr,
                // })
                element_ptr
            }
            spmt::Expression::ConstructExtern { t, args } => {
                // ConstructExtern is like Construct but with named arguments.
                // Naga's Compose doesn't use names, so we extract just the values.
                let naga_ty = self.type_cache.convert_type_full(
                    &mut self.module.borrow_mut(),
                    self.converter.extern_converter,
                    t,
                );
                let mut components = Vec::new();
                for (_name, arg) in args {
                    components.push(self.convert_expression(arg));
                }
                self.append_and_emit(Expression::Compose {
                    ty: naga_ty,
                    components,
                })
            }
            spmt::Expression::ArrayLiteral(expressions) => {
                // ArrayLiteral creates an array from a list of expressions.
                // We need to infer the element type from the first expression if possible.
                if expressions.is_empty() {
                    panic!("Cannot create an array literal with no elements");
                }

                // Convert all expressions
                let mut components = Vec::new();
                for expr in expressions {
                    components.push(self.convert_expression(expr));
                }

                // Try to derive the element type from the first expression
                let element_ty_spmt = self
                    .try_derive_type(&expressions[0])
                    .expect("Cannot infer array element type");
                let element_ty = {
                    let mut module = self.module.borrow_mut();
                    self.type_cache.convert_type_full(
                        &mut module,
                        self.converter.extern_converter,
                        &element_ty_spmt,
                    )
                };
                let stride = {
                    let module = self.module.borrow();
                    match module.types[element_ty].inner {
                        TypeInner::Scalar(s) => s.width as u32,
                        TypeInner::Vector { scalar, size } => scalar.width as u32 * size as u32,
                        TypeInner::Struct { span, .. } => span,
                        _ => 0,
                    }
                };

                // Create an array type
                let array_size = naga::ArraySize::Constant(
                    core::num::NonZeroU32::new(expressions.len() as u32)
                        .expect("Array size must be non-zero"),
                );

                let array_ty = self.module.borrow_mut().types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Array {
                            base: element_ty,
                            size: array_size,
                            stride,
                        },
                    },
                    Span::UNDEFINED,
                );

                self.append_and_emit(Expression::Compose {
                    ty: array_ty,
                    components,
                })
            }
        };

        // Cache the converted expression for potential reuse
        self.already_converted.push((expr.clone(), r));
        r
    }

    /// Convert a known math function call to a naga Math expression.
    fn convert_math_call(
        &mut self,
        math_fn: MathFunction,
        parameters: &[spmt::Expression<'a>],
        _parameter_types: &[spmt::VariableType],
    ) -> Handle<Expression> {
        let arg = self.convert_expression(&parameters[0]);
        let arg1 = parameters.get(1).map(|p| self.convert_expression(p));
        let arg2 = parameters.get(2).map(|p| self.convert_expression(p));
        let arg3 = parameters.get(3).map(|p| self.convert_expression(p));

        self.append_and_emit(Expression::Math {
            fun: math_fn,
            arg,
            arg1,
            arg2,
            arg3,
        })
    }

    /// Optionally cast an expression if needed for the target parameter type.
    fn maybe_cast(
        &mut self,
        expr: Handle<Expression>,
        _target_type: &spmt::VariableType,
    ) -> Handle<Expression> {
        // For now, no casting needed — types are matched at construction.
        // In the future, this could handle f64→f32 demotion etc.
        expr
    }

    /// Compute a default 1D index from pos3 for density variable access.
    /// index = (pos3.x * dims.y + pos3.y) * dims.z + pos3.z
    pub(crate) fn convert_default_density_index(
        &mut self,
        dim_y: Handle<Expression>,
        dim_z: Handle<Expression>,
    ) -> Handle<Expression> {
        // We need to find the pos3 variable — it's always argument 0 for density functions.
        let pos3 = self.append_and_emit(Expression::FunctionArgument(0));

        let x = self.append_and_emit(Expression::AccessIndex {
            base: pos3,
            index: 0,
        });
        let y = self.append_and_emit(Expression::AccessIndex {
            base: pos3,
            index: 1,
        });
        let z = self.append_and_emit(Expression::AccessIndex {
            base: pos3,
            index: 2,
        });

        // x * dims.y
        let x_times_dy = self.append_and_emit(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: x,
            right: dim_y,
        });

        // x * dims.y + y
        let xy = self.append_and_emit(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: x_times_dy,
            right: y,
        });

        // (x * dims.y + y) * dims.z
        let xy_times_dz = self.append_and_emit(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: xy,
            right: dim_z,
        });

        // (x * dims.y + y) * dims.z + z
        self.append_and_emit(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: xy_times_dz,
            right: z,
        })
    }

    // tries to convert arguments of a binary operator to match the expected types for the operator
    // for example vec3<i32> + vec3<f32> would convert the i32 argument to f32 before emitting the binary operator

    pub fn try_convert_arguments_binary_op(
        &mut self,
        left: &spmt::Expression<'a>,
        right: &spmt::Expression<'a>,
        op: &spmt::BinaryOperator,
    ) -> (Handle<Expression>, Handle<Expression>) {
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
                spmt::VariableType::Pos3,
                spmt::VariableType::Vec3,
            ) => {
                let converted_left = self.convert_pos3_to_vec3(left_h);
                (converted_left, right_h)
            }
            (
                spmt::BinaryOperator::Add
                | spmt::BinaryOperator::Subtract
                | spmt::BinaryOperator::Multiply
                | spmt::BinaryOperator::Divide,
                spmt::VariableType::Vec3,
                spmt::VariableType::Pos3,
            ) => {
                let converted_right = self.convert_pos3_to_vec3(right_h);
                (left_h, converted_right)
            }

            (
                spmt::BinaryOperator::Add
                | spmt::BinaryOperator::Subtract
                | spmt::BinaryOperator::Multiply
                | spmt::BinaryOperator::Divide,
                spmt::VariableType::I32,
                spmt::VariableType::Bool,
            ) => {
                // Convert bool to i32 (false -> 0, true -> 1)
                let converted_right = self.append_and_emit(Expression::Compose {
                    ty: self.type_cache.u32_ty,
                    components: vec![right_h],
                });
                (left_h, converted_right)
            }
            _ => (left_h, right_h), // No conversion needed or possible
        }
    }

    fn convert_pos3_to_vec3(&mut self, pos3_expr: Handle<Expression>) -> Handle<Expression> {
        // vec3<f32>(pos3)
        let vec3_ty = self.type_cache.vec3f_ty;
        self.append_and_emit(Expression::Compose {
            ty: vec3_ty,
            components: vec![pos3_expr],
        })
    }

    fn try_derive_type(&self, expr: &spmt::Expression<'a>) -> Option<spmt::VariableType> {
        match expr {
            spmt::Expression::Variable(var) => Some(var.t.clone()),
            spmt::Expression::Float(_) => Some(spmt::VariableType::F32),
            spmt::Expression::Double(_) => Some(spmt::VariableType::F64),
            spmt::Expression::Int(_) => Some(spmt::VariableType::I32),
            spmt::Expression::Long(_) => Some(spmt::VariableType::I64),
            spmt::Expression::BinaryOp { op, left, right } => {
                let left_type = self.try_derive_type(left)?;
                let right_type = self.try_derive_type(right)?;
                self.try_derive_binop_type(*op, left_type, right_type)
            }
            spmt::Expression::UnaryOp { operand, .. } => self.try_derive_type(operand),
            spmt::Expression::Field { .. } => None,
            spmt::Expression::FunctionCall { .. } => None, // Could be derived from function signature if needed
            spmt::Expression::ExternCall { .. } => None, // Could be derived from extern declaration if needed
            spmt::Expression::DensityVariable(_, _) => Some(spmt::VariableType::DensityInput),
            spmt::Expression::PermutationTable(_) => Some(spmt::VariableType::PermutationTable),
            spmt::Expression::Construct { t, .. } => Some(t.clone()),
            spmt::Expression::ArrayAccess { array, index } => {
                let array_type = self.try_derive_type(array)?;
                match array_type {
                    spmt::VariableType::Vec3 => Some(spmt::VariableType::F32), // Accessing a vec3 component gives f32
                    spmt::VariableType::Pos3 => Some(spmt::VariableType::I32),
                    // spmt::VariableType::Array(element_type, _) => Some(match element_type {
                    //     "f32" | "f64" => spmt::VariableType::F32,
                    //     "i32" => spmt::VariableType::I32,
                    //     "i64" => spmt::VariableType::I64,
                    //     "bool" => spmt::VariableType::Bool,
                    //     name => spmt::VariableType::Extern(name),
                    // }),
                    spmt::VariableType::Array(element_type, _) => Some(*element_type),
                    _ => None, // For other array types, we would need more information to derive the element type
                }
            }
            spmt::Expression::ConstructExtern { t, .. } => Some(t.clone()),
            spmt::Expression::ArrayLiteral(expressions) => {
                if expressions.is_empty() {
                    return None;
                }
                // Try to derive the element type from the first expression
                let element_type = self.try_derive_type(&expressions[0])?;
                Some(spmt::VariableType::Array(
                    // match element_type {
                    //     spmt::VariableType::F32 => "f32",
                    //     spmt::VariableType::I32 => "i32",
                    //     spmt::VariableType::I64 => "i64",
                    //     spmt::VariableType::Bool => "bool",
                    //     spmt::VariableType::Extern(name) => name,
                    //     _ => return None, // Cannot represent complex types as array element type strings
                    // },
                    Box::new(element_type),
                    expressions.len(),
                ))
            }
        }
    }

    fn try_derive_binop_type(
        &self,
        op: spmt::BinaryOperator,
        left_type: spmt::VariableType,
        right_type: spmt::VariableType,
    ) -> Option<spmt::VariableType> {
        match op {
            spmt::BinaryOperator::Add
            | spmt::BinaryOperator::Subtract
            | spmt::BinaryOperator::Multiply
            | spmt::BinaryOperator::Divide => {
                match (&left_type, &right_type) {
                    (spmt::VariableType::Vec3, spmt::VariableType::Pos3)
                    | (spmt::VariableType::Pos3, spmt::VariableType::Vec3) => {
                        Some(spmt::VariableType::Vec3)
                    }
                    _ if left_type == right_type => Some(left_type),
                    _ => None, // Types don't match and we don't know how to convert
                }
            }
            _ => None, // Other operators not handled for type conversion
        }
    }
}

/// Try to map a known extern function name to a Naga built-in MathFunction.
fn try_map_math_function(name: &str) -> Option<MathFunction> {
    match name {
        "abs" => Some(MathFunction::Abs),
        "min" => Some(MathFunction::Min),
        "max" => Some(MathFunction::Max),
        "clamp" => Some(MathFunction::Clamp),
        "floor" => Some(MathFunction::Floor),
        "ceil" => Some(MathFunction::Ceil),
        "fract" => Some(MathFunction::Fract),
        "sqrt" => Some(MathFunction::Sqrt),
        "sign" => Some(MathFunction::Sign),
        "pow" => Some(MathFunction::Pow),
        "sin" => Some(MathFunction::Sin),
        "cos" => Some(MathFunction::Cos),
        "mix" => Some(MathFunction::Mix),
        "step" => Some(MathFunction::Step),
        "smoothstep" => Some(MathFunction::SmoothStep),
        _ => None,
    }
}
