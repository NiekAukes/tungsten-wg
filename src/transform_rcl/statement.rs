/*
Statement conversion from SPMT to RCL.
Handles conversion of all statement types including assignments,
returns, control flow (if/while), and expression statements.
*/

use super::{RCLFunctionConverter, expression};
use crate::rcl::model as rcl;
use crate::spmt::model::{self as spmt, Addr};

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
        }
    }
}
