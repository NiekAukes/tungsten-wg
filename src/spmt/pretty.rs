use std::{
    cell::Cell,
    collections::HashSet,
    fmt::{Write, format},
    rc::Rc,
};

use crate::spmt::model::{
    Addr, DensityFunction, Expression, Function, Interned, SPMT, Statement, Variable,
};

const INDENT: &str = "    ";

pub struct Printer {
    out: String,
    line: String,
    indent: usize,
    name_cache: std::collections::HashMap<*const (), String>,
    anon_counter: usize,
}

impl Printer {
    pub fn new() -> Self {
        Self {
            out: String::new(),
            line: String::new(),
            indent: 0,
            anon_counter: 0,
            name_cache: std::collections::HashMap::new(),
        }
    }

    pub fn new_with_name_cache(name_cache: std::collections::HashMap<*const (), String>) -> Self {
        Self {
            out: String::new(),
            line: String::new(),
            indent: 0,
            anon_counter: 0,
            name_cache: name_cache,
        }
    }

    pub fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str(INDENT);
        }
        self.out.push_str(&self.line);
        self.line.clear();
        self.out.push_str(s);
        self.out.push('\n');
    }

    pub fn push(&mut self, s: &str) {
        self.line.push_str(s);
    }

    pub fn indent(&mut self) {
        self.indent += 1;
    }

    pub fn dedent(&mut self) {
        self.indent -= 1;
    }

    pub fn finish(self) -> String {
        self.finish_with_name_cache().0
    }

    pub fn finish_with_name_cache(self) -> (String, std::collections::HashMap<*const (), String>) {
        (self.out + &self.line, self.name_cache)
    }

    pub fn anon_name<T: Addr>(&mut self, entity: T, prefix: &str) -> String {
        if let Some(name) = self.name_cache.get(&(entity.addr())) {
            return name.clone();
        }

        let name = format!("<{}-{}>", prefix, self.anon_counter);
        self.anon_counter += 1;
        self.name_cache.insert(entity.addr(), name.clone());
        name
    }
}

pub trait PrettyPrint {
    fn pretty(&self, p: &mut Printer);
}

/* =========================
Top-level SPMT
========================= */
impl<'m> PrettyPrint for SPMT<'m> {
    fn pretty(&self, p: &mut Printer) {
        for f in &self.functions {
            f.pretty(p);
            p.line("");
        }

        for d in &self.density_functions {
            d.pretty(p);
            p.line("");
        }
    }
}

/* =========================
Functions
========================= */
impl<'m> PrettyPrint for Function<'m> {
    fn pretty(&self, p: &mut Printer) {
        p.push("fn ");
        if let Some(canonical_name) = self.canonical_name.as_deref() {
            p.push(canonical_name);
        } else {
            let name = p.anon_name(self, "function");

            p.push(&name);
        };
        p.push("(");

        for (i, param) in self.parameters.iter().enumerate() {
            if i > 0 {
                p.push(", ");
            }
            let param_name = param.name.clone().unwrap_or(p.anon_name(param, "param"));
            write!(p.line, "{}: {:?}", param_name, param.t).unwrap();
        }

        p.line(") {");
        p.indent();

        for stmt in &self.body {
            stmt.pretty(p);
        }

        p.dedent();
        p.line("}");
    }
}

impl<'m> PrettyPrint for DensityFunction<'m> {
    fn pretty(&self, p: &mut Printer) {
        p.push("density ");

        if let Some(canonical_name) = self.canonical_name.as_deref() {
            p.push(canonical_name);
        } else {
            let name = p.anon_name(self, "density-function");
            p.push(&name);
        };

        p.line("(p: Vec3) {");
        p.indent();

        if !self.density_inputs.is_empty() {
            p.line("// density inputs:");
            for d in &self.density_inputs {
                let n = p.anon_name(d.density_function, "density-function");
                p.line(&format!("// - {}", n));
            }
            p.line("");
        }

        for stmt in &self.body {
            stmt.pretty(p);
        }

        p.dedent();
        p.line("}");
    }
}

/* =========================
Statements
========================= */
impl<'m> PrettyPrint for Statement<'m> {
    fn pretty(&self, p: &mut Printer) {
        match self {
            Statement::Assign { target, value } => {
                let target_name = target.name.clone().unwrap_or(p.anon_name(target, "var"));
                p.push(&target_name);
                p.push(" = ");
                value.pretty(p);
                p.line(";");
            }

            Statement::Return(expr) => {
                p.push("return ");
                expr.pretty(p);
                p.line(";");
            }

            Statement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                p.push("if ");
                condition.pretty(p);
                p.line(" {");
                p.indent();

                for s in then_branch {
                    s.pretty(p);
                }

                p.dedent();

                if !else_branch.is_empty() {
                    p.line("} else {");
                    p.indent();

                    for s in else_branch {
                        s.pretty(p);
                    }

                    p.dedent();
                }

                p.line("}");
            }

            Statement::While { condition, body } => {
                p.push("while ");
                condition.pretty(p);
                p.line(" {");
                p.indent();

                for s in body {
                    s.pretty(p);
                }

                p.dedent();
                p.line("}");
            }
        }
    }
}

/* =========================
Expressions
========================= */
impl<'m> PrettyPrint for Expression<'m> {
    fn pretty(&self, p: &mut Printer) {
        match self {
            Expression::Variable(v) => {
                let name = v.name.clone().unwrap_or(p.anon_name(v.clone(), "var"));
                p.push(&name);
            }
            Expression::Float(v) => {
                write!(p.line, "{:.6}", v).unwrap();
            }
            Expression::Int(v) => {
                write!(p.line, "{}", v).unwrap();
            }
            Expression::Long(v) => {
                write!(p.line, "{}", v).unwrap();
            }
            Expression::FunctionCall {
                function,
                parameters,
            } => {
                // let name = function
                //     .canonical_name
                //     .as_deref()
                //     .unwrap_or_else(|| p.anon_name(function, "func"))
                //     .to_string();
                if let Some(name) = function.canonical_name.as_deref() {
                    p.push(name);
                } else {
                    let name = p.anon_name(*function, "func");
                    p.push(&name);
                }
                p.push("(");

                for (i, param) in parameters.iter().enumerate() {
                    if i > 0 {
                        p.push(", ");
                    }
                    param.pretty(p);
                }

                p.push(")");
            }
            Expression::ExternCall {
                function_name,
                parameters,
                parameter_types: _,
            } => {
                p.push("extern ");
                p.push(function_name);
                p.push("(");
                for (i, param) in parameters.iter().enumerate() {
                    if i > 0 {
                        p.push(", ");
                    }
                    param.pretty(p);
                }
                p.push(")");
            }
            Expression::BinaryOp { op, left, right } => {
                p.push("(");
                left.pretty(p);
                p.push(match op {
                    crate::spmt::model::BinaryOperator::Add => " + ",
                    crate::spmt::model::BinaryOperator::Subtract => " - ",
                    crate::spmt::model::BinaryOperator::Multiply => " * ",
                    crate::spmt::model::BinaryOperator::Divide => " / ",
                    // super::model::BinaryOperator::Add => todo!(),
                    // super::model::BinaryOperator::Subtract => todo!(),
                    // super::model::BinaryOperator::Multiply => todo!(),
                    // super::model::BinaryOperator::Divide => todo!(),
                    // super::model::BinaryOperator::Equal => todo!(),
                    // super::model::BinaryOperator::NotEqual => todo!(),
                    // super::model::BinaryOperator::Less => todo!(),
                    // super::model::BinaryOperator::LessEqual => todo!(),
                    // super::model::BinaryOperator::Greater => todo!(),
                    // super::model::BinaryOperator::GreaterEqual => todo!(),
                    // super::model::BinaryOperator::And => todo!(),
                    // super::model::BinaryOperator::Or => todo!(),
                    crate::spmt::model::BinaryOperator::Equal => " == ",
                    crate::spmt::model::BinaryOperator::NotEqual => " != ",
                    crate::spmt::model::BinaryOperator::Less => " < ",
                    crate::spmt::model::BinaryOperator::LessEqual => " <= ",
                    crate::spmt::model::BinaryOperator::Greater => " > ",
                    crate::spmt::model::BinaryOperator::GreaterEqual => " >= ",
                    crate::spmt::model::BinaryOperator::And => " && ",
                    crate::spmt::model::BinaryOperator::Or => " || ",
                });
                right.pretty(p);
                p.push(")");
            }
            Expression::DensityVariable(density_input) => {
                let name = p.anon_name(density_input.density_function, "density-function");
                p.push(&name);
            }
            Expression::PermutationTable(perm_table_input) => {
                let name = format!(
                    "perm-table-{}-{}",
                    perm_table_input.ident,
                    perm_table_input
                        .subident
                        .as_ref()
                        .unwrap_or(&"".to_string())
                );
                p.push(&name);
            }
            Expression::UnaryOp { op, operand } => {
                p.push(match op {
                    crate::spmt::model::UnaryOperator::Negate => "-",
                });
                operand.pretty(p);
            }
            Expression::Field { base, field } => {
                base.pretty(p);
                p.push(".");
                p.push(field);
            }
            // Expression::MakeVec3 { x, y, z } => {
            //     p.push("vec3(");
            //     x.pretty(p);
            //     p.push(", ");
            //     y.pretty(p);
            //     p.push(", ");
            //     z.pretty(p);
            //     p.push(")");
            // }
            Expression::Construct { t, args } => {
                p.push(&format!("{:?}(", t));
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        p.push(", ");
                    }
                    arg.pretty(p);
                }
                p.push(")");
            }
        }
    }
}

impl<'m> DensityFunction<'m> {
    pub fn pretty_with_deps(&self, p: &mut Printer) {
        // Print helper function dependencies first
        if !self.helper_functions.is_empty() {
            p.line("// Helper function dependencies:");
            for func_ref in &self.helper_functions {
                func_ref.pretty(p);
                p.line(""); // Add spacing between helper functions
            }
        }

        // Print the density function itself
        self.pretty(p);
    }
}
