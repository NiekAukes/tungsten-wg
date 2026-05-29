/*
CUDA C++ code generation from the CUDA model.

This module generates CUDA C++ source code from the CudaModule intermediate
representation.
*/

use super::model as cuda;
use crate::spmt::pretty::Printer;

/// Generates CUDA C++ source code from a `CudaModule`.
pub struct CudaCodeGenerator;

impl CudaCodeGenerator {
    pub fn new() -> Self {
        CudaCodeGenerator
    }

    /// Generate a complete `.cu` file from the given module.
    pub fn generate_module(&self, module: &cuda::CudaModule<'_>) -> String {
        let mut p = Printer::new();

        p.line("// Auto-generated CUDA C++ module");
        p.line("#pragma once");
        p.line("");

        // Includes
        for inc in &module.includes {
            p.push("#include ");
            p.line(inc);
        }
        if !module.includes.is_empty() {
            p.line("");
        }

        // Struct definitions
        for s in &module.structs {
            self.generate_struct(&mut p, s);
            p.line("");
        }

        // Global / constant variables
        for gv in &module.global_vars {
            self.generate_global_var(&mut p, gv);
        }
        if !module.global_vars.is_empty() {
            p.line("");
        }

        // Device helper functions
        for f in &module.device_functions {
            self.generate_function(&mut p, f);
            p.line("");
        }

        // Kernel entry-points
        for k in &module.kernels {
            self.generate_function(&mut p, k);
            p.line("");
        }

        p.finish()
    }

    // -----------------------------------------------------------------------
    // Struct
    // -----------------------------------------------------------------------

    fn generate_struct(&self, p: &mut Printer, s: &cuda::StructRef<'_>) {
        p.push("struct ");
        p.push(&s.name);
        p.line(" {");
        p.indent();
        for (field_name, field_type) in &s.fields {
            p.push(&self.type_to_string(field_type));
            p.push(" ");
            p.push(field_name);
            p.line(";");
        }
        p.dedent();
        p.line("};");
    }

    // -----------------------------------------------------------------------
    // Global variable
    // -----------------------------------------------------------------------

    fn generate_global_var(&self, p: &mut Printer, gv: &cuda::GlobalVar<'_>) {
        if let Some(q) = &gv.qualifier {
            p.push(&format!("{} ", q));
        }
        p.push(&self.type_to_string(&gv.t));
        p.push(" ");
        p.push(&gv.name);
        if let Some(init) = &gv.init {
            p.push(" = ");
            let init_str = self.expression_to_string(p, init);
            p.push(&init_str);
        }
        p.line(";");
    }

    // -----------------------------------------------------------------------
    // Function
    // -----------------------------------------------------------------------

    fn generate_function(&self, p: &mut Printer, func: &cuda::FunctionRef<'_>) {
        // extern "C"
        if func.is_extern_c {
            p.line("extern \"C\" {");
            p.indent();
        }

        // Template
        if !func.template_params.is_empty() {
            p.push("template <");
            p.push(&func.template_params.join(", "));
            p.line(">");
        }

        // Qualifier + inline
        if func.is_inline {
            p.push("inline ");
        }
        p.push(&format!("{} ", func.qualifier));

        // Return type
        p.push(&self.type_to_string(&func.return_type));
        p.push(" ");

        // Name
        if let Some(name) = &func.name {
            p.push(name);
        } else {
            let anon = sanitize_name(&p.anon_name(*func, "func"));
            p.push(&anon);
        }

        // Parameters
        p.push("(");
        for (i, param) in func.parameters.iter().enumerate() {
            if i > 0 {
                p.push(", ");
            }
            if param.is_const
                && !matches!(
                    param.t,
                    cuda::Type::Pointer(_) | cuda::Type::ConstPointer(_)
                )
            {
                p.push("const ");
            }
            p.push(&self.type_to_string(&param.t));
            p.push(" ");
            p.push(&param.name);
        }
        p.line(") {");

        p.indent();

        // Local variable declarations
        for v in &func.variables {
            self.generate_local_var_decl(p, v);
        }

        // Body
        for stmt in &func.body {
            self.generate_statement(p, stmt);
        }

        p.dedent();
        p.line("}");

        if func.is_extern_c {
            p.dedent();
            p.line("}");
        }
    }

    fn generate_local_var_decl(&self, p: &mut Printer, v: &std::rc::Rc<cuda::Variable>) {
        if let Some(q) = &v.memory_qualifier {
            p.push(&format!("{} ", q));
        }
        let name = match &v.name {
            crate::spmt::model::Name::Anonymous => sanitize_name(&p.anon_name(v.clone(), "var")),
            crate::spmt::model::Name::Prefixed(prefix) => {
                sanitize_name(&p.anon_name(v.clone(), &prefix))
            }
            crate::spmt::model::Name::Named(n) => sanitize_name(n),
        };
        p.push(&self.type_decl_string(&v.t, &name));
        p.line(";");
    }

    // -----------------------------------------------------------------------
    // Statements
    // -----------------------------------------------------------------------

    fn generate_statement(&self, p: &mut Printer, stmt: &cuda::Statement<'_>) {
        match stmt {
            cuda::Statement::Assign { target, value } => {
                let var_name = self.variable_name(target, p);
                p.push(&var_name);
                p.push(" = ");
                let val_str = self.expression_to_string(p, value);
                p.push(&val_str);
                p.line(";");
            }

            cuda::Statement::ArrayAssign {
                target,
                index,
                value,
            } => {
                let var_name = self.variable_name(target, p);
                p.push(&var_name);
                p.push("[");
                let idx = self.expression_to_string(p, index);
                p.push(&idx);
                p.push("] = ");
                let val = self.expression_to_string(p, value);
                p.push(&val);
                p.line(";");
            }

            cuda::Statement::Return(Some(expr)) => {
                p.push("return ");
                let expr_str = self.expression_to_string(p, expr);
                p.push(&expr_str);
                p.line(";");
            }
            cuda::Statement::Return(None) => {
                p.line("return;");
            }

            cuda::Statement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                p.push("if (");
                let cond = self.expression_to_string(p, condition);
                p.push(&cond);
                p.line(") {");
                p.indent();
                for s in then_branch {
                    self.generate_statement(p, s);
                }
                p.dedent();
                if let Some(else_stmts) = else_branch {
                    p.line("} else {");
                    p.indent();
                    for s in else_stmts {
                        self.generate_statement(p, s);
                    }
                    p.dedent();
                    p.line("}");
                } else {
                    p.line("}");
                }
            }

            cuda::Statement::While { condition, body } => {
                p.push("while (");
                let cond = self.expression_to_string(p, condition);
                p.push(&cond);
                p.line(") {");
                p.indent();
                for s in body {
                    self.generate_statement(p, s);
                }
                p.dedent();
                p.line("}");
            }

            cuda::Statement::For {
                init,
                condition,
                increment,
                body,
            } => {
                p.push("for (");
                if let Some(init_stmt) = init {
                    // Strip the trailing newline by generating into a sub-printer
                    let init_str = self.stmt_to_inline_string(p, init_stmt);
                    p.push(&init_str);
                }
                p.push("; ");
                if let Some(cond_expr) = condition {
                    let cond = self.expression_to_string(p, cond_expr);
                    p.push(&cond);
                }
                p.push("; ");
                if let Some(incr_stmt) = increment {
                    let incr = self.stmt_to_inline_string(p, incr_stmt);
                    p.push(&incr);
                }
                p.line(") {");
                p.indent();
                for s in body {
                    self.generate_statement(p, s);
                }
                p.dedent();
                p.line("}");
            }

            cuda::Statement::Declare {
                variable,
                init,
                is_const,
            } => {
                if let Some(q) = &variable.memory_qualifier {
                    p.push(&format!("{} ", q));
                }
                if *is_const {
                    p.push("const ");
                }
                let name = self.variable_name(variable, p);
                p.push(&self.type_decl_string(&variable.t, &name));
                if let Some(expr) = init {
                    p.push(" = ");
                    let val = self.expression_to_string(p, expr);
                    p.push(&val);
                }
                p.line(";");
            }

            cuda::Statement::ExprStatement(expr) => {
                let expr_str = self.expression_to_string(p, expr);
                p.push(&expr_str);
                p.line(";");
            }

            cuda::Statement::Block(stmts) => {
                p.line("{");
                p.indent();
                for s in stmts {
                    self.generate_statement(p, s);
                }
                p.dedent();
                p.line("}");
            }

            cuda::Statement::SyncThreads => {
                p.line("__syncthreads();");
            }

            cuda::Statement::Break => {
                p.line("break;");
            }

            cuda::Statement::InlineCuda(s) => {
                p.line(s);
            }
        }
    }

    /// Render a statement as a single-line string (used inside `for` headers).
    /// Strips the trailing semicolon so the caller can insert separators.
    fn stmt_to_inline_string(&self, p: &mut Printer, stmt: &cuda::Statement<'_>) -> String {
        match stmt {
            cuda::Statement::Assign { target, value } => {
                let name = self.variable_name(target, p);
                let val = self.expression_to_string(p, value);
                format!("{} = {}", name, val)
            }
            cuda::Statement::Declare {
                variable,
                init,
                is_const,
            } => {
                let qualifier = variable
                    .memory_qualifier
                    .map(|q| format!("{} ", q))
                    .unwrap_or_default();
                let const_prefix = if *is_const { "const " } else { "" };
                let name = self.variable_name(variable, p);
                let decl = self.type_decl_string(&variable.t, &name);
                let init_str = init
                    .as_ref()
                    .map(|e| format!(" = {}", self.expression_to_string(p, e)))
                    .unwrap_or_default();
                format!("{}{}{}{}", qualifier, const_prefix, decl, init_str)
            }
            _ => {
                // Fallback for other statement types — emit normally and strip newline
                "/* complex for-init */".to_string()
            }
        }
    }

    // -----------------------------------------------------------------------
    // Expressions
    // -----------------------------------------------------------------------

    fn expression_to_string(&self, p: &mut Printer, expr: &cuda::Expression<'_>) -> String {
        match expr {
            cuda::Expression::Variable(v) => self.variable_name(v, p),
            cuda::Expression::I32Literal(v) => format!("{}", v),
            cuda::Expression::U32Literal(v) => format!("{}u", v),
            cuda::Expression::I64Literal(v) => format!("{}LL", v),
            cuda::Expression::U64Literal(v) => format!("{}ULL", v),
            cuda::Expression::F32Literal(v) => {
                // if float is infinity or NaN, we need to use the special literals
                if v.is_infinite() {
                    if v.is_sign_positive() {
                        "INFINITY".to_string()
                    } else {
                        "-INFINITY".to_string()
                    }
                } else if v.is_nan() {
                    "NAN".to_string()
                } else {
                    format!("{}f", v)
                }
            }
            cuda::Expression::F64Literal(v) => {
                // if float is infinity or NaN, we need to use the special literals
                if v.is_infinite() {
                    if v.is_sign_positive() {
                        "INFINITY".to_string()
                    } else {
                        "-INFINITY".to_string()
                    }
                } else if v.is_nan() {
                    "NAN".to_string()
                } else {
                    format!("{}", v)
                }
            }
            cuda::Expression::BoolLiteral(v) => format!("{}", v),

            cuda::Expression::BinaryOp { op, left, right } => {
                format!(
                    "({} {} {})",
                    self.expression_to_string(p, left),
                    self.binary_op_to_str(*op),
                    self.expression_to_string(p, right)
                )
            }
            cuda::Expression::UnaryOp { op, operand } => {
                format!(
                    "({}{})",
                    self.unary_op_to_str(*op),
                    self.expression_to_string(p, operand)
                )
            }

            cuda::Expression::FunctionCall {
                function,
                arguments,
            } => {
                let name = if let Some(n) = &function.name {
                    n.clone()
                } else {
                    sanitize_name(&p.anon_name(*function, "func"))
                };
                let args: Vec<String> = arguments
                    .iter()
                    .map(|a| self.expression_to_string(p, a))
                    .collect();
                format!("{}({})", name, args.join(", "))
            }
            cuda::Expression::LateBoundCall {
                function_name,
                arguments,
            } => {
                let args: Vec<String> = arguments
                    .iter()
                    .map(|a| self.expression_to_string(p, a))
                    .collect();
                format!("{}({})", function_name, args.join(", "))
            }

            cuda::Expression::Field { base, field } => {
                format!("{}.{}", self.expression_to_string(p, base), field)
            }
            cuda::Expression::Index { base, index } => {
                format!(
                    "{}[{}]",
                    self.expression_to_string(p, base),
                    self.expression_to_string(p, index)
                )
            }
            cuda::Expression::Deref(inner) => {
                format!("(*{})", self.expression_to_string(p, inner))
            }
            cuda::Expression::AddressOf(inner) => {
                format!("(&{})", self.expression_to_string(p, inner))
            }
            cuda::Expression::Cast { expr, to_type } => {
                format!(
                    "(({})({})",
                    self.type_to_string(to_type),
                    self.expression_to_string(p, expr)
                )
            }

            cuda::Expression::StructInit {
                struct_name,
                fields,
            } => {
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|(name, e)| format!(".{} = {}", name, self.expression_to_string(p, e)))
                    .collect();
                format!("({}){{ {} }}", struct_name, field_strs.join(", "))
            }

            cuda::Expression::ThreadIdx(axis) => format!("threadIdx.{}", axis),
            cuda::Expression::BlockIdx(axis) => format!("blockIdx.{}", axis),
            cuda::Expression::BlockDim(axis) => format!("blockDim.{}", axis),
            cuda::Expression::GridDim(axis) => format!("gridDim.{}", axis),

            cuda::Expression::ArrayLiteral(exprs) => {
                let items: Vec<String> = exprs
                    .iter()
                    .map(|e| self.expression_to_string(p, e))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }

            cuda::Expression::InlineCuda(s) => s.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn variable_name(&self, v: &std::rc::Rc<cuda::Variable>, p: &mut Printer) -> String {
        match &v.name {
            crate::spmt::model::Name::Named(n) => sanitize_name(n),
            crate::spmt::model::Name::Prefixed(prefix) => {
                sanitize_name(&p.anon_name(v.clone(), prefix))
            }
            crate::spmt::model::Name::Anonymous => sanitize_name(&p.anon_name(v.clone(), "var")),
        }
    }

    fn type_to_string(&self, t: &cuda::Type) -> String {
        format!("{}", t)
    }

    /// Generate the C/C++ declarator string for a variable of type `t` named `name`.
    /// For array types this must be `elem_type name[N]` rather than `elem_type[N] name`.
    fn type_decl_string(&self, t: &cuda::Type, name: &str) -> String {
        match t {
            cuda::Type::Array(inner, n) => format!("{} {}[{}]", inner, name, n),
            other => format!("{} {}", other, name),
        }
    }

    fn binary_op_to_str(&self, op: cuda::BinaryOperator) -> &'static str {
        match op {
            cuda::BinaryOperator::Add => "+",
            cuda::BinaryOperator::Subtract => "-",
            cuda::BinaryOperator::Multiply => "*",
            cuda::BinaryOperator::Divide => "/",
            cuda::BinaryOperator::Modulo => "%",
            cuda::BinaryOperator::Equal => "==",
            cuda::BinaryOperator::NotEqual => "!=",
            cuda::BinaryOperator::Less => "<",
            cuda::BinaryOperator::LessEqual => "<=",
            cuda::BinaryOperator::Greater => ">",
            cuda::BinaryOperator::GreaterEqual => ">=",
            cuda::BinaryOperator::And => "&&",
            cuda::BinaryOperator::Or => "||",
            cuda::BinaryOperator::BitwiseAnd => "&",
            cuda::BinaryOperator::BitwiseOr => "|",
            cuda::BinaryOperator::BitwiseXor => "^",
            cuda::BinaryOperator::LeftShift => "<<",
            cuda::BinaryOperator::RightShift => ">>",
        }
    }

    fn unary_op_to_str(&self, op: cuda::UnaryOperator) -> &'static str {
        match op {
            cuda::UnaryOperator::Negate => "-",
            cuda::UnaryOperator::Not => "!",
            cuda::UnaryOperator::BitwiseNot => "~",
        }
    }
}

// ---------------------------------------------------------------------------
// Name sanitisation
// ---------------------------------------------------------------------------

/// Turn a printer-generated anonymous name (e.g. `<func-0>`) into a valid
/// C++ identifier.
fn sanitize_name(name: &str) -> String {
    name.replace(':', "_")
        .replace('/', "_")
        .replace('<', "_")
        .replace('>', "_")
        .replace('-', "_")
}

// ---------------------------------------------------------------------------
// Public convenience entry-point
// ---------------------------------------------------------------------------

pub fn generate_cuda_module(module: &cuda::CudaModule<'_>) -> String {
    CudaCodeGenerator::new().generate_module(module)
}
