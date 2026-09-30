use bumpalo::Bump;

use crate::spmt::{builder::{collector::Collector, hash::Fingerprint}, model::*};
use std::{
    cell::RefCell,
    collections::{hash_map::DefaultHasher, HashSet},
    hash::{Hash, Hasher},
};

type E<'m> = Expression<'m>;
type S<'m> = Statement<'m>;

mod hash;
mod collector;
pub mod macros;
pub use macros::*;

#[derive(Default)]
struct State<'m> {
    locals: Vec<Var<'m>>,
    constants: Vec<(Var<'m>, Expression<'m>)>,
    /// Every density input created via `density()`, used to dedupe. Only the ones
    /// actually referenced by the body end up in the output.
    density_cache: Vec<DensityInput<'m>>,
}

pub struct Builder<'m> {
    arena: &'m Bump,
    canonical_name: Option<String>,
    params: Vec<Var<'m>>,
    return_type: VariableType,
    body: Vec<Statement<'m>>,
    source_hash: Option<u64>,
    state: RefCell<State<'m>>,
}

impl<'m> DensityInput<'m> {
    /// Read the density at the current position.
    pub fn get(&self) -> Expression<'m> {
        Expression::DensityVariable(self.clone(), None)
    }
    /// Read the density at an explicit index.
    pub fn at(&self, index: impl Into<Expression<'m>>) -> Expression<'m> {
        Expression::DensityVariable(self.clone(), Some(Box::new(index.into())))
    }
}

impl<'m> Builder<'m> {
    pub fn new(arena: &'m Bump) -> Self {
        Self {
            arena,
            canonical_name: None,
            params: Vec::new(),
            return_type: VariableType::F32,
            body: Vec::new(),
            source_hash: None,
            state: RefCell::new(State::default()),
        }
    }

    // ---- configuration -------------------------------------------------

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.canonical_name = Some(name.into());
        self
    }

    /// Provide a hash of the *source* (e.g. of the input JSON) instead of
    /// deriving one from the IR structure.
    pub fn with_source_hash(mut self, hash: u64) -> Self {
        self.source_hash = Some(hash);
        self
    }

    pub fn returns(mut self, t: VariableType) -> Self {
        self.return_type = t;
        self
    }

    pub fn set_body(&mut self, body: Vec<Statement<'m>>) {
        self.body = body;
    }

    // ---- variables -----------------------------------------------------

    fn alloc_var(&self, name: Name, t: VariableType) -> Var<'m> {
        Interned::new(self.arena.alloc(Variable { name, t }))
    }

    fn named_var(&self, st: &State<'m>, name: &str, t: VariableType) -> Var<'m> {
        let taken = self
            .params
            .iter()
            .chain(&st.locals)
            .chain(st.constants.iter().map(|(v, _)| v))
            .any(|v| matches!(&v.name, Name::Named(n) if n == name));
        assert!(!taken, "variable `{name}` is already declared");
        self.alloc_var(Name::Named(name.to_owned()), t)
    }

    /// Function parameter (only meaningful for `build_function`).
    pub fn param(&mut self, name: &str, t: VariableType) -> Var<'m> {
        let v = {
            let st = self.state.borrow();
            self.named_var(&st, name, t)
        };
        self.params.push(v);
        v
    }

    /// Uniquely named local. This is what `spmt_body!`'s `let` expands to.
    pub fn local(&self, name: &str, t: VariableType) -> Var<'m> {
        let mut st = self.state.borrow_mut();
        let v = self.named_var(&st, name, t);
        st.locals.push(v);
        v
    }

    /// Compiler temporary. The name is only a hint, so no uniqueness check.
    pub fn temp(&self, prefix: &str, t: VariableType) -> Var<'m> {
        let v = self.alloc_var(Name::Prefixed(prefix.to_owned()), t);
        self.state.borrow_mut().locals.push(v);
        v
    }

    /// A named constant, hoisted out of the body.
    pub fn constant(
        &self,
        name: &str,
        t: VariableType,
        value: impl Into<Expression<'m>>,
    ) -> Var<'m> {
        let mut st = self.state.borrow_mut();
        let v = self.named_var(&st, name, t);
        st.constants.push((v, value.into()));
        v
    }

    // ---- density inputs ------------------------------------------------

    /// Declare a dependency on another density function. Identical requests
    /// return the same input (and therefore the same variable).
    pub fn density(
        &self,
        func: DensityFunctionRef<'m>,
        scaled_origin: (f64, f64, f64),
        scaled_position: (f64, f64, f64),
        dimensions: (i32, i32, i32),
    ) -> DensityInput<'m> {
        let mut st = self.state.borrow_mut();
        if let Some(existing) = st.density_cache.iter().find(|d| {
            d.density_function == func
                && d.scaled_origin == scaled_origin
                && d.scaled_position == scaled_position
                && d.dimensions == dimensions
        }) {
            return existing.clone();
        }
        let n = st.density_cache.len();
        let var = self.alloc_var(
            Name::Named(format!("in_density_{n}")),
            VariableType::DensityInput,
        );
        let input = DensityInput { var, density_function: func, scaled_origin, scaled_position, dimensions };
        st.density_cache.push(input.clone());
        input
    }

    // ---- finishing -----------------------------------------------------

    pub fn build_function(self) -> Function<'m> {
        Function {
            canonical_name: self.canonical_name,
            parameters: self.params,
            body: self.body,
            variables: self.state.into_inner().locals,
            return_type: self.return_type,
        }
    }

    /// Same, but interned in the arena so it can be used in `FunctionCall`.
    pub fn finish_function(self) -> FunctionRef<'m> {
        let arena = self.arena;
        Interned::new(arena.alloc(self.build_function()))
    }

    pub fn build_compute_unit(self) -> DensityFunction<'m> {
        let State { locals, constants, .. } = self.state.into_inner();

        // Walk constants first, then the body, so the order is stable.
        let mut c = Collector::default();
        for (_, value) in &constants {
            c.expr(value);
        }
        c.stmts(&self.body);
        let Collector { helpers, density_inputs, mut perm_tables, .. } = c;

        // Deterministic order for the permutation tables (that's what `Ord` is for).
        perm_tables.sort();
        perm_tables.dedup();

        let source_hash = self.source_hash.unwrap_or_else(|| {
            let mut h = Fingerprint(DefaultHasher::new());
            self.canonical_name.hash(&mut h.0);
            for (v, e) in &constants {
                h.var(v);
                h.expr(e);
            }
            for d in &density_inputs {
                h.density_input(d);
            }
            perm_tables.hash(&mut h.0);
            for v in &locals {
                h.var(v);
            }
            h.stmts(&self.body);
            h.0.finish()
        });

        DensityFunction {
            canonical_name: self.canonical_name,
            density_inputs,
            permutation_table_inputs: perm_tables,
            body: self.body,
            variables: locals,
            helper_functions: helpers,
            constants,
            source_hash,
        }
    }

    pub fn finish_compute_unit(self) -> DensityFunctionRef<'m> {
        let arena = self.arena;
        Interned::new(arena.alloc(self.build_compute_unit()))
    }
}

