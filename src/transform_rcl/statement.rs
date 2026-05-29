/*
Statement conversion from SPMT to RCL.
Handles conversion of all statement types including assignments,
returns, control flow (if/while), and expression statements.
*/

use std::rc::Rc;

use super::{RCLFunctionConverter, expression};
use crate::orchestrate::Scale;
use crate::rcl::model as rcl;
use crate::spmt::model::{self as spmt, Addr};
use crate::transform_rcl::InputKey;

/// Convert an SPMT statement to an RCL statement

impl<'a, 'm> RCLFunctionConverter<'m> {
    pub fn convert_statement(&mut self, stmt: &spmt::Statement<'a>) -> rcl::Statement<'m> {
        match stmt {
            spmt::Statement::Assign { target, value } => {
                let target_var = self.get_or_create_variable(target.clone());
                let value_expr = self.convert_expression(value);
                rcl::Statement::Assign {
                    target: target_var,
                    value: value_expr,
                }
            }
            spmt::Statement::Return(expr) => {
                let expr = self.convert_expression(expr);
                rcl::Statement::Return(Some(expr))
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
                rcl::Statement::If {
                    condition,
                    then_branch,
                    else_branch,
                }
            }
            spmt::Statement::While { condition, body } => {
                let condition = self.convert_expression(condition);
                let body = body.iter().map(|s| self.convert_statement(s)).collect();
                rcl::Statement::While { condition, body }
            }
            spmt::Statement::Repeat { count, body } => {
                // essentially the same as: for i in 0..count { body }
                let counter_var = Rc::new(rcl::Variable {
                    name: Some("repeat_counter".to_string()),
                    t: rcl::Type::I32,
                    mutable: true,
                });
                self.add_raw_variable(
                    InputKey {
                        density_function: body.addr(),
                        dimensions: (0, 0, 0),
                        scaled_origin: Scale::default(),
                    },
                    counter_var.clone(),
                );
                let init_stmt = rcl::Statement::Declare {
                    variable: counter_var.clone(),
                    init: Some(rcl::Expression::I32Literal(0)),
                    mutable: true,
                };
                let condition = rcl::Expression::BinaryOp {
                    op: rcl::BinaryOperator::Less,
                    left: Box::new(rcl::Expression::Variable(counter_var.clone())),
                    right: Box::new(rcl::Expression::I32Literal(*count as i32)),
                };
                let mut body: Vec<rcl::Statement<'_>> =
                    body.iter().map(|s| self.convert_statement(s)).collect();

                // Increment counter at the end of the loop body
                body.push(rcl::Statement::Assign {
                    target: counter_var.clone(),
                    value: rcl::Expression::BinaryOp {
                        op: rcl::BinaryOperator::Add,
                        left: Box::new(rcl::Expression::Variable(counter_var.clone())),
                        right: Box::new(rcl::Expression::I32Literal(1)),
                    },
                });

                let mut before_loop = vec![init_stmt];
                before_loop.push(rcl::Statement::While { condition, body });
                rcl::Statement::Block(before_loop)
            }
            spmt::Statement::Break => rcl::Statement::Break,
        }
    }
}
