/*
Extern function declaration generation for Naga IR.
When SPMT code calls extern functions (perlin, hermite, interpolate484, etc.),
these are generated as Naga function declarations with matching signatures
but placeholder bodies (returning zero).

Known math builtins (abs, min, max, clamp) are mapped to naga::MathFunction
in expression.rs instead — they don't need extern declarations.
*/

use std::{
    cell::{Cell, RefCell, RefMut},
    collections::HashMap,
};

use naga::{
    Arena, Block, Constant, Expression, Function, FunctionArgument, FunctionResult, Handle,
    Literal, Span, Statement,
};

use super::types::TypeCache;
use crate::spmt::model as spmt;

#[derive(Debug)]
pub struct ExternFunctionConverter<'a> {
    helper_module: &'a naga::Module,

    converted_functions: RefCell<HashMap<Handle<Function>, Handle<Function>>>,
    converted_types: RefCell<HashMap<Handle<naga::Type>, Handle<naga::Type>>>,
    converted_constants: RefCell<HashMap<Handle<naga::Constant>, Handle<naga::Constant>>>,
    converted_global_expressions:
        RefCell<HashMap<Handle<naga::Expression>, Handle<naga::Expression>>>,
}

impl<'a> ExternFunctionConverter<'a> {
    pub fn new(helper_module: &'a naga::Module) -> Self {
        ExternFunctionConverter {
            helper_module,
            converted_functions: RefCell::new(HashMap::new()),
            converted_types: RefCell::new(HashMap::new()),
            converted_constants: RefCell::new(HashMap::new()),
            converted_global_expressions: RefCell::new(HashMap::new()),
        }
    }

    fn create_stub_extern_function(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        name: &str,
        argument_types: Vec<Handle<naga::Type>>,
        return_type: Option<Handle<naga::Type>>,
    ) -> Handle<Function> {
        let arguments = argument_types
            .into_iter()
            .map(|ty| FunctionArgument {
                name: None,
                ty,
                binding: None,
            })
            .collect();

        let result = return_type.map(|ty| FunctionResult { ty, binding: None });

        let func = Function {
            name: Some(super::types::sanitize_name(name)),
            arguments,
            result,
            ..Default::default()
        };

        module.functions.append(func, Span::UNDEFINED)
    }

    /// Parse a WGSL file once (at compile time via include_str!) and cache the parsed module.
    /// Then copy the named function into the target module, remapping all type/expression handles.
    pub fn embed_wgsl_function(
        &self,
        mut module: RefMut<'_, naga::Module>,
        function_name: &str,
    ) -> Handle<Function> {
        // Find the function by name
        let (handle, _) = self
            .helper_module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some(function_name))
            .unwrap_or_else(|| {
                panic!(
                    "Helper function '{}' not found in helper module!",
                    function_name
                );
            });

        self.embed_wgsl_function_by_handle(&mut module, handle)
    }

    fn embed_wgsl_function_by_handle(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        function_handle: Handle<Function>,
    ) -> Handle<Function> {
        // if the function was already copied, return the existing handle
        if let Some(&existing) = self.converted_functions.borrow().get(&function_handle) {
            return existing;
        }

        // Copy types from helper_module into module, building a handle remap
        // Then copy the function body with remapped handles
        self.copy_function_into_module(module, function_handle)
    }

    fn copy_function_into_module(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        src_func: Handle<Function>,
    ) -> Handle<Function> {
        // if the function was already copied, return the existing handle
        if let Some(&existing) = self.converted_functions.borrow_mut().get(&src_func) {
            return existing;
        }

        let src_function = &self.helper_module.functions[src_func];
        let dst_args = src_function
            .arguments
            .iter()
            .map(|arg| naga::FunctionArgument {
                name: arg.name.as_ref().map(|n| super::types::sanitize_name(n)),
                ty: self.copy_type_with_remap(module, arg.ty),
                binding: arg.binding.clone(),
            })
            .collect();
        let dst_result = src_function.result.as_ref().map(|res| FunctionResult {
            ty: self.copy_type_with_remap(module, res.ty),
            binding: res.binding.clone(),
        });

        let mut dst_function = Function {
            name: src_function
                .name
                .as_ref()
                .map(|n| super::types::sanitize_name(n)),
            arguments: dst_args,
            result: dst_result,
            local_variables: src_function.local_variables.clone(),
            expressions: src_function.expressions.clone(),
            body: src_function.body.clone(),
            named_expressions: src_function.named_expressions.clone(),
            diagnostic_filter_leaf: src_function.diagnostic_filter_leaf,
        };

        self.remap_local_variable(module, &mut dst_function.local_variables);
        self.remap_expressions(module, &mut dst_function.expressions);
        self.remap_statement(module, &mut dst_function.body);

        let dst_handle = module.functions.append(dst_function, Span::UNDEFINED);
        self.converted_functions
            .borrow_mut()
            .insert(src_func, dst_handle);
        dst_handle
    }

    pub fn copy_type_with_remap(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        src: Handle<naga::Type>,
    ) -> Handle<naga::Type> {
        // if the type was already copied, return the existing handle
        if let Some(&existing) = self.converted_types.borrow().get(&src) {
            return existing;
        }

        let src_ty = &self.helper_module.types[src];
        let new_inner = match &src_ty.inner {
            naga::TypeInner::Scalar(scalar) => naga::TypeInner::Scalar(*scalar),
            naga::TypeInner::Vector { size, scalar } => naga::TypeInner::Vector {
                size: *size,
                scalar: scalar.clone(),
            },
            naga::TypeInner::Array { base, size, stride } => {
                let base = self.copy_type_with_remap(module, *base);
                naga::TypeInner::Array {
                    base,
                    size: *size,
                    stride: *stride,
                }
            }
            naga::TypeInner::Struct { members, span } => {
                let members = members
                    .iter()
                    .map(|m| naga::StructMember {
                        name: m.name.clone(),
                        ty: self.copy_type_with_remap(module, m.ty),
                        binding: m.binding.clone(),
                        offset: m.offset,
                    })
                    .collect();
                naga::TypeInner::Struct {
                    members,
                    span: *span,
                }
            }
            _ => unimplemented!("Unsupported type inner for copying: {:?}", src_ty.inner),
        };
        let t = naga::Type {
            name: src_ty.name.clone(),
            inner: new_inner,
        };
        let h = module.types.insert(t, Span::UNDEFINED);
        self.converted_types.borrow_mut().insert(src, h);
        h
    }

    pub fn remap_expressions(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        expressions: &mut Arena<Expression>,
    ) {
        for (_, expr) in expressions.iter_mut() {
            // match expr {
            //     Expression::CallResult(c) => {
            //         let new_func = self.copy_function_into_module(module, *c);
            //         *c = new_func;
            //     }
            //     Expression::Compose { ty, components: _ } => {
            //         let new_ty = self.copy_type_with_remap(module, *ty);
            //         *ty = new_ty;
            //     }
            //     Expression::Constant(h) => {
            //         let new_const = self.copy_constant_with_remap(module, *h);
            //         *h = new_const;
            //     }
            //     _ => {}
            // }
            self.remap_expression(module, expr);
        }
    }

    fn remap_expression(&self, module: &'a mut RefMut<'_, naga::Module>, expr: &mut Expression) {
        match expr {
            Expression::CallResult(c) => {
                let new_func = self.copy_function_into_module(module, *c);
                *c = new_func;
            }
            Expression::Compose { ty, components: _ } => {
                let new_ty = self.copy_type_with_remap(module, *ty);
                *ty = new_ty;
            }
            Expression::Constant(h) => {
                let new_const = self.copy_constant_with_remap(module, *h);
                *h = new_const;
            }
            _ => {}
        }
    }

    pub fn remap_local_variable(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        local_variables: &mut Arena<naga::LocalVariable>,
    ) {
        for (_, var) in local_variables.iter_mut() {
            let new_ty = self.copy_type_with_remap(module, var.ty);
            var.ty = new_ty;
        }
    }

    pub fn remap_statement(&self, module: &'a mut RefMut<'_, naga::Module>, stmts: &mut Block) {
        for stmt in stmts.iter_mut() {
            match stmt {
                Statement::Call {
                    function,
                    arguments: _,
                    result: _,
                } => {
                    let new_func = self.copy_function_into_module(module, *function);
                    *function = new_func;
                }
                _ => {}
            }
        }
    }

    pub fn copy_constant_with_remap(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        src: Handle<naga::Constant>,
    ) -> Handle<naga::Constant> {
        if let Some(&existing) = self.converted_constants.borrow().get(&src) {
            return existing;
        }

        let src_const = &self.helper_module.constants[src];
        let new_ty = self.copy_type_with_remap(module, src_const.ty);
        let new_init = self.copy_const_expression(module, src_const.init);
        let new_const = Constant {
            name: src_const.name.clone(),
            ty: new_ty,
            init: new_init,
        };
        let handle = module.constants.append(new_const, Span::UNDEFINED);
        self.converted_constants.borrow_mut().insert(src, handle);
        handle
    }

    pub fn copy_const_expression(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        src: Handle<naga::Expression>,
    ) -> Handle<naga::Expression> {
        if let Some(&existing) = self.converted_global_expressions.borrow().get(&src) {
            return existing;
        }

        let src_expr = &self.helper_module.global_expressions[src];
        let new_expr = match src_expr {
            Expression::Literal(literal) => Expression::Literal(*literal),
            Expression::Constant(handle) => {
                Expression::Constant(self.copy_constant_with_remap(module, *handle))
            }
            Expression::Override(handle) => Expression::Override(*handle),
            Expression::ZeroValue(handle) => {
                Expression::ZeroValue(self.copy_type_with_remap(module, *handle))
            }
            Expression::Compose { ty, components } => Expression::Compose {
                ty: self.copy_type_with_remap(module, *ty),
                components: components
                    .iter()
                    .map(|component| self.copy_const_expression(module, *component))
                    .collect(),
            },
            Expression::Access { base, index } => Expression::Access {
                base: self.copy_const_expression(module, *base),
                index: self.copy_const_expression(module, *index),
            },
            Expression::AccessIndex { base, index } => Expression::AccessIndex {
                base: self.copy_const_expression(module, *base),
                index: *index,
            },
            Expression::Splat { size, value } => Expression::Splat {
                size: *size,
                value: self.copy_const_expression(module, *value),
            },
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => Expression::Swizzle {
                size: *size,
                vector: self.copy_const_expression(module, *vector),
                pattern: *pattern,
            },
            Expression::Unary { op, expr } => Expression::Unary {
                op: *op,
                expr: self.copy_const_expression(module, *expr),
            },
            Expression::Binary { op, left, right } => Expression::Binary {
                op: *op,
                left: self.copy_const_expression(module, *left),
                right: self.copy_const_expression(module, *right),
            },
            Expression::Select {
                condition,
                accept,
                reject,
            } => Expression::Select {
                condition: self.copy_const_expression(module, *condition),
                accept: self.copy_const_expression(module, *accept),
                reject: self.copy_const_expression(module, *reject),
            },
            Expression::Relational { fun, argument } => Expression::Relational {
                fun: *fun,
                argument: self.copy_const_expression(module, *argument),
            },
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => Expression::Math {
                fun: *fun,
                arg: self.copy_const_expression(module, *arg),
                arg1: arg1.map(|handle| self.copy_const_expression(module, handle)),
                arg2: arg2.map(|handle| self.copy_const_expression(module, handle)),
                arg3: arg3.map(|handle| self.copy_const_expression(module, handle)),
            },
            Expression::As {
                expr,
                kind,
                convert,
            } => Expression::As {
                expr: self.copy_const_expression(module, *expr),
                kind: *kind,
                convert: *convert,
            },
            Expression::FunctionArgument(_)
            | Expression::GlobalVariable(_)
            | Expression::LocalVariable(_)
            | Expression::Load { .. }
            | Expression::ImageSample { .. }
            | Expression::ImageLoad { .. }
            | Expression::ImageQuery { .. }
            | Expression::Derivative { .. }
            | Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::ArrayLength(_)
            | Expression::RayQueryVertexPositions { .. }
            | Expression::RayQueryProceedResult
            | Expression::RayQueryGetIntersection { .. }
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. }
            | Expression::CooperativeLoad { .. }
            | Expression::CooperativeMultiplyAdd { .. } => {
                panic!(
                    "Unsupported const expression in helper module: {:?}",
                    src_expr
                )
            }
        };

        let handle = module.global_expressions.append(new_expr, Span::UNDEFINED);
        self.converted_global_expressions
            .borrow_mut()
            .insert(src, handle);
        handle
    }

    pub fn copy_expression_with_remap(
        &self,
        module: &'a mut RefMut<'_, naga::Module>,
        src: Handle<naga::Expression>,
    ) -> Handle<naga::Expression> {
        self.copy_const_expression(module, src)
    }

    //
    pub fn embed_wgsl_struct(
        &self,
        module: &mut RefMut<'_, naga::Module>,
        name: &str,
    ) -> Handle<naga::Type> {
        // Find the struct by name
        let (handle, _) = self
            .helper_module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| {
                panic!("Helper struct '{}' not found in helper module!", name);
            });
        self.copy_type_with_remap(module, handle)
    }
}
