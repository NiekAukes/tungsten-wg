/*
Statement conversion from SPMT to CUDA C++.
*/

use std::rc::Rc;

use super::CudaFunctionConverter;
use crate::cuda::model as cuda;
use crate::spmt::model as spmt;

impl<'a, 'm> CudaFunctionConverter<'m> {
    pub fn convert_statement(&mut self, stmt: &spmt::Statement<'a>) -> cuda::Statement<'m> {
        match stmt {
            spmt::Statement::Assign {
                target,
                value: spmt::Expression::ArrayLiteral(x),
            } => {
                // array literals need to be converted into a series of assignments to the array elements, since CUDA C++ doesn't support array literals.
                let target_var = self.get_or_create_variable(target.clone());
                let mut statements = Vec::new();
                for (i, elem) in x.iter().enumerate() {
                    let assign = cuda::Statement::ArrayAssign {
                        target: target_var.clone(),
                        index: cuda::Expression::I32Literal(i as i32),
                        value: self.convert_expression(elem),
                    };
                    statements.push(assign);
                }
                // Wrap the assignments in a block to ensure they execute together.
                cuda::Statement::Block(statements)
            }
            spmt::Statement::Assign { target, value } => {
                let target_var = self.get_or_create_variable(target.clone());
                let value_expr = self.convert_expression(value);
                cuda::Statement::Assign {
                    target: target_var,
                    value: value_expr,
                }
            }
            spmt::Statement::Return(expr) => {
                let expr = self.convert_expression(expr);
                cuda::Statement::Return(Some(expr))
            }
            spmt::Statement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition = self.convert_expression(condition);
                let then_branch = then_branch
                    .iter()
                    .map(|s| self.convert_statement(s))
                    .collect();
                let else_branch = if else_branch.is_empty() {
                    None
                } else {
                    Some(
                        else_branch
                            .iter()
                            .map(|s| self.convert_statement(s))
                            .collect(),
                    )
                };
                cuda::Statement::If {
                    condition,
                    then_branch,
                    else_branch,
                }
            }
            spmt::Statement::While { condition, body } => {
                let condition = self.convert_expression(condition);
                let body = body.iter().map(|s| self.convert_statement(s)).collect();
                cuda::Statement::While { condition, body }
            }
            spmt::Statement::Repeat { count, body } => {
                // `for (int __repeat_i = 0; __repeat_i < count; ++__repeat_i) { body }`
                let counter_var = Rc::new(cuda::Variable {
                    name: spmt::Name::Named("__repeat_i".to_string()),
                    t: cuda::Type::Int32,
                    memory_qualifier: None,
                });
                let init = cuda::Statement::Declare {
                    variable: counter_var.clone(),
                    init: Some(cuda::Expression::I32Literal(0)),
                    is_const: false,
                };
                let condition = cuda::Expression::BinaryOp {
                    op: cuda::BinaryOperator::Less,
                    left: Box::new(cuda::Expression::Variable(counter_var.clone())),
                    right: Box::new(cuda::Expression::I32Literal(*count as i32)),
                };
                let increment = cuda::Statement::Assign {
                    target: counter_var.clone(),
                    value: cuda::Expression::BinaryOp {
                        op: cuda::BinaryOperator::Add,
                        left: Box::new(cuda::Expression::Variable(counter_var)),
                        right: Box::new(cuda::Expression::I32Literal(1)),
                    },
                };
                let body = body.iter().map(|s| self.convert_statement(s)).collect();
                cuda::Statement::For {
                    init: Some(Box::new(init)),
                    condition: Some(condition),
                    increment: Some(Box::new(increment)),
                    body,
                }
            }
            spmt::Statement::Break => cuda::Statement::Break,
        }
    }
}
