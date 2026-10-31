//! Bottom-up base-type synthesis, used **only** to pick a `FieldSel`. It never fills an
//! annotation. If it is wrong, the lowered `Field` has a
//! wrong selector and the checker rejects it.

use std::collections::{HashMap, HashSet};

use tungsten_ir::{
    AmbientKind, BinaryOperator, ComputeUnit, ConstExpr, Constant, DensityInput, Expression,
    Function, HostInput, OptimizedProgram, PermutationTableInput, Ref, VariableType,
};

use crate::lower::get;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fail {
    /// No rule gives a type (empty array literal, index into a non-array, constant cycle).
    Unsynthesizable,
    /// A ref doesn't resolve. The lowering reports that on its own, so no second error.
    Broken,
}

pub(crate) type Synth = Result<VariableType, Fail>;

pub(crate) struct Synthesizer<'p> {
    p: &'p OptimizedProgram,
    /// Density input → its source unit.
    sources: HashMap<Ref<DensityInput>, Ref<ComputeUnit>>,
    memo: HashMap<Ref<Constant>, Synth>,
    in_progress: HashSet<Ref<Constant>>,
}

impl<'p> Synthesizer<'p> {
    pub(crate) fn new(p: &'p OptimizedProgram) -> Self {
        let mut sources = HashMap::new();
        for (_, u) in p.compute_units.iter() {
            let ComputeUnit::DensityFunction { density_inputs, .. } = u;
            for (r, d) in density_inputs.iter() {
                sources.insert(r, d.density_function);
            }
        }
        Self {
            p,
            sources,
            memo: HashMap::new(),
            in_progress: HashSet::new(),
        }
    }

    fn constant(&mut self, r: Ref<Constant>) -> Synth {
        if let Some(s) = self.memo.get(&r) {
            return s.clone();
        }
        if !self.in_progress.insert(r) {
            return Err(Fail::Unsynthesizable);
        }
        let s = match get(&self.p.constants, r) {
            Some(c) => self.expr(None, &const_to_expr(&c.value)),
            None => Err(Fail::Broken),
        };
        self.in_progress.remove(&r);
        self.memo.insert(r, s.clone());
        s
    }

    /// `f` is the enclosing function (`None` in a constant initializer).
    pub(crate) fn expr(&mut self, f: Option<&Function>, e: &Expression) -> Synth {
        use VariableType as T;
        Ok(match e {
            Expression::Constant(r) => return self.constant(*r),
            Expression::Variable(r) => f
                .and_then(|f| get(&f.variables, *r))
                .map(|v| v.t.clone())
                .ok_or(Fail::Broken)?,
            Expression::Float(_) => T::F32,
            Expression::Double(_) => T::F64,
            Expression::Byte(_) => T::I8,
            Expression::Int(_) => T::I32,
            Expression::Long(_) => T::I64,
            Expression::Bool(_) => T::Bool,
            Expression::FunctionCall { function, .. } => get(&self.p.functions, *function)
                .map(|f| f.return_type.clone())
                .ok_or(Fail::Broken)?,
            Expression::ExternCall { return_type, .. } => return_type.clone(),
            Expression::DensityVariable(r, _) => {
                let unit = self.sources.get(r).ok_or(Fail::Broken)?;
                let ComputeUnit::DensityFunction { function, .. } =
                    get(&self.p.compute_units, *unit).ok_or(Fail::Broken)?;
                get(&self.p.functions, *function)
                    .ok_or(Fail::Broken)?
                    .return_type
                    .clone()
            }
            Expression::PermutationTable(r) => match get(&self.p.permutation_tables, *r) {
                Some(PermutationTableInput::PerlinNoise { .. }) => T::PermutationTable,
                Some(PermutationTableInput::Base3DNoise) => T::InterpolatedNoiseSampler,
                None => return Err(Fail::Broken),
            },
            Expression::HostInput(r) => match get(&self.p.host_inputs, *r) {
                Some(HostInput::Static { t, .. }) => t.clone(),
                // Dynamic hosts are reported at the host.
                Some(HostInput::Dynamic) | None => return Err(Fail::Broken),
            },
            Expression::Ambient(AmbientKind::Pos3) => T::Pos3,
            Expression::Ambient(_) => T::Vec3,
            Expression::BinaryOp { op, left, right } => {
                use BinaryOperator as B;
                match op {
                    B::Equal
                    | B::NotEqual
                    | B::Less
                    | B::LessEqual
                    | B::Greater
                    | B::GreaterEqual
                    | B::And
                    | B::Or => T::Bool,
                    B::Add | B::Subtract | B::Multiply | B::Divide => {
                        let l = self.expr(f, left)?;
                        let r = self.expr(f, right);
                        match (&l, r) {
                            (T::Vec3, Ok(T::Pos3)) | (T::Pos3, Ok(T::Vec3)) => T::Vec3,
                            _ => l,
                        }
                    }
                    _ => self.expr(f, left)?,
                }
            }
            Expression::UnaryOp { operand, .. } => return self.expr(f, operand),
            Expression::Field { type_of_field, .. } => type_of_field.clone(),
            Expression::ArrayAccess { array, .. } => match self.expr(f, array)? {
                T::Array(elem, _) => *elem,
                T::Vec3 => T::F64,
                T::Pos3 => T::I32,
                _ => return Err(Fail::Unsynthesizable),
            },
            Expression::Construct { t, .. } | Expression::ConstructExtern { t, .. } => t.clone(),
            Expression::ArrayLiteral(elems) => match elems.first() {
                Some(first) => T::Array(Box::new(self.expr(f, first)?), elems.len()),
                None => return Err(Fail::Unsynthesizable),
            },
            Expression::ExplicitCast { to, .. } => to.clone(),
        })
    }
}

/// `ConstExpr` is a subset of `Expression`; constants lower and synthesize through this.
pub(crate) fn const_to_expr(c: &ConstExpr) -> Expression {
    let b = |e: &ConstExpr| Box::new(const_to_expr(e));
    match c {
        ConstExpr::Constant(r) => Expression::Constant(*r),
        ConstExpr::Float(v) => Expression::Float(*v),
        ConstExpr::Double(v) => Expression::Double(*v),
        ConstExpr::Byte(v) => Expression::Byte(*v),
        ConstExpr::Int(v) => Expression::Int(*v),
        ConstExpr::Long(v) => Expression::Long(*v),
        ConstExpr::Bool(v) => Expression::Bool(*v),
        ConstExpr::BinaryOp { op, left, right } => Expression::BinaryOp {
            op: *op,
            left: b(left),
            right: b(right),
        },
        ConstExpr::UnaryOp { op, operand } => Expression::UnaryOp {
            op: *op,
            operand: b(operand),
        },
        ConstExpr::ExplicitCast { to, expr } => Expression::ExplicitCast {
            to: to.clone(),
            expr: b(expr),
        },
        ConstExpr::Field {
            base,
            field,
            type_of_field,
            known_idnex,
        } => Expression::Field {
            base: b(base),
            field: field.clone(),
            type_of_field: type_of_field.clone(),
            known_idnex: *known_idnex,
        },
        ConstExpr::ArrayAccess { array, index } => Expression::ArrayAccess {
            array: b(array),
            index: b(index),
        },
        ConstExpr::ArrayLiteral(elems) => {
            Expression::ArrayLiteral(elems.iter().map(const_to_expr).collect())
        }
        ConstExpr::Construct { t, args } => Expression::Construct {
            t: t.clone(),
            args: args.iter().map(const_to_expr).collect(),
        },
    }
}
