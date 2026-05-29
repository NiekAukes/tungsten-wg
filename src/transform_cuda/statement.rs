/*
Statement conversion from SPMT to CUDA C++.
*/

use super::CudaFunctionConverter;
use crate::cuda::model as cuda;
use crate::spmt::model as spmt;

impl<'a, 'm> CudaFunctionConverter<'m> {
    pub fn convert_statement(&mut self, stmt: &spmt::Statement<'a>) -> cuda::Statement<'m> {
        match stmt {
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
        }
    }
}
