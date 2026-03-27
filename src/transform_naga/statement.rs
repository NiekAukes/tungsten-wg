/*
Statement conversion from SPMT to Naga IR.
Handles conversion of all statement types: assignments (Store),
returns, control flow (If, Loop for while).
*/

use naga::{Block, Expression, GlobalVariable, Handle, Span, Statement};

use super::InputKey;
use super::expression::ExprContext;
use crate::spmt::model as spmt;

/// Convert an SPMT statement to Naga statements, appending them to the given block.
/// This is separate from ExprContext because statements may need to manipulate
/// the block being built (e.g., if/while create sub-blocks).
pub fn convert_statement<'a>(stmt: &spmt::Statement<'a>, ctx: &mut ExprContext<'_, 'a, '_>) {
    match stmt {
        spmt::Statement::Assign { target, value } => {
            let value_h = ctx.convert_expression(value);

            let key = InputKey::from(*target);
            let ptr = if let Some(&existing) = ctx.converter.var_map.get(&key) {
                existing
            } else {
                // Variable not yet registered — create a local variable for it.
                let naga_ty = ctx.type_cache.convert_type(&target.t);
                let local_handle = ctx.func.local_variables.append(
                    naga::LocalVariable {
                        name: target
                            .name
                            .as_deref()
                            .map(|n| super::types::sanitize_name(n)),
                        ty: naga_ty,
                        init: None,
                    },
                    Span::UNDEFINED,
                );
                let ptr_expr = ctx
                    .func
                    .expressions
                    .append(Expression::LocalVariable(local_handle), Span::UNDEFINED);
                ctx.converter.var_map.insert(key, ptr_expr);
                ptr_expr
            };

            ctx.func.body.push(
                Statement::Store {
                    pointer: ptr,
                    value: value_h,
                },
                Span::UNDEFINED,
            );
        }

        spmt::Statement::Return(expr) => {
            let value_h = ctx.convert_expression(expr);
            ctx.func.body.push(
                Statement::Return {
                    value: Some(value_h),
                },
                Span::UNDEFINED,
            );
        }

        spmt::Statement::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_h = ctx.convert_expression(condition);

            // Build accept block
            // We need to temporarily swap out the body to build sub-blocks,
            // then swap back. This is the standard pattern for building nested blocks in naga.
            let saved_body = std::mem::replace(&mut ctx.func.body, Block::new());

            for s in then_branch {
                convert_statement(s, ctx);
            }
            let accept = std::mem::replace(&mut ctx.func.body, Block::new());

            // Build reject block
            for s in else_branch {
                convert_statement(s, ctx);
            }
            let reject = std::mem::replace(&mut ctx.func.body, saved_body);

            ctx.func.body.push(
                Statement::If {
                    condition: cond_h,
                    accept,
                    reject,
                },
                Span::UNDEFINED,
            );
        }

        spmt::Statement::While { condition, body } => {
            // Naga uses Loop { body, continuing, break_if }.
            // We implement `while (cond) { body }` as:
            //   loop {
            //     if (!cond) { break; }
            //     body...
            //   }
            let saved_body = std::mem::replace(&mut ctx.func.body, Block::new());

            // Evaluate condition and negate it for the break check
            let cond_h = ctx.convert_expression(condition);
            let old_len = ctx.func.expressions.len();
            let not_cond = ctx.func.expressions.append(
                Expression::Unary {
                    op: naga::UnaryOperator::LogicalNot,
                    expr: cond_h,
                },
                Span::UNDEFINED,
            );
            let not_cond_range = ctx.func.expressions.range_from(old_len);
            ctx.func
                .body
                .push(Statement::Emit(not_cond_range), Span::UNDEFINED);

            // if (!cond) { break; }
            let mut break_block = Block::new();
            break_block.push(Statement::Break, Span::UNDEFINED);
            ctx.func.body.push(
                Statement::If {
                    condition: not_cond,
                    accept: break_block,
                    reject: Block::new(),
                },
                Span::UNDEFINED,
            );

            // Convert loop body statements
            for s in body {
                convert_statement(s, ctx);
            }

            let loop_body = std::mem::replace(&mut ctx.func.body, saved_body);

            ctx.func.body.push(
                Statement::Loop {
                    body: loop_body,
                    continuing: Block::new(),
                    break_if: None,
                },
                Span::UNDEFINED,
            );
        }
    }
}

pub fn convert_density_return_statement<'a>(
    stmt: &spmt::Statement<'a>,
    output_handle: Handle<GlobalVariable>,
    pos3_idx: u32,
    dimensions_handle: Handle<GlobalVariable>,
    ctx: &mut ExprContext<'_, 'a, '_>,
) {
    // output[gid] = value;
    let spmt::Statement::Return(expr) = stmt else {
        panic!("Expected return statement");
    };
    let value_h = ctx.convert_expression(expr);

    // Get the global variable handle for the output buffer

    let pos3 = ctx
        .func
        .expressions
        .append(Expression::FunctionArgument(pos3_idx), Span::UNDEFINED);

    // Compute the index expression (global invocation ID)
    let dim_expr = ctx.func.expressions.append(
        Expression::GlobalVariable(dimensions_handle),
        Span::UNDEFINED,
    );
    let dim_expr = ctx.append_and_emit(Expression::Load { pointer: dim_expr });
    let dim_y = ctx.append_and_emit(Expression::AccessIndex {
        base: dim_expr,
        index: 1,
    });
    let dim_z = ctx.append_and_emit(Expression::AccessIndex {
        base: dim_expr,
        index: 2,
    });
    let index_expr = ctx.convert_default_density_index(dim_y, dim_z);

    let base_expr = ctx.append_and_emit(Expression::GlobalVariable(output_handle));
    //let base_expr = ctx.append_and_emit(Expression::Load { pointer: base_expr });

    // Compute the pointer to output[gid]
    let ptr_expr = ctx.append_and_emit(Expression::Access {
        base: base_expr,
        index: index_expr,
    });

    // Store the return value into output[gid]
    ctx.func.body.push(
        Statement::Store {
            pointer: ptr_expr,
            value: value_h,
        },
        Span::UNDEFINED,
    );
}
