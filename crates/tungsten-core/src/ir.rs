//! Ids, types, literals, expressions, statements, functions, units and the program.
//!
//! All fields are `pub` and there are no exec functions: any exec fn that indexes a `Vec`
//! inside `verus!{}` would need bounds proofs. Plain-Rust helpers live in `tungsten-check`.

use vstd::prelude::*;

use crate::registry::{ExternDecl, RecordDecl};

verus! {

// ---------------------------------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------------------------------

/// Index into `Program.units`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnitId(pub usize);

/// Index into `Program.functions`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FnId(pub usize);

/// Index into `Program.consts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstId(pub usize);

/// Index into `Program.hosts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HostId(pub usize);

/// Index into `Program.tables`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TableId(pub usize);

/// Index into `Program.inputs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InputId(pub usize);

/// Index into `Program.externs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExternId(pub usize);

/// Index into `Program.records`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecordId(pub usize);

/// Index into the enclosing `Function.locals`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalId(pub usize);

// ---------------------------------------------------------------------------------------------
// Types and literals
// ---------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum Type {
    Bool,
    I8,
    I32,
    I64,
    F32,
    F64,
    Vec3,
    Pos3,
    Array(Box<Type>, usize),
    /// From `VariableType::Extern(name)`.
    Record(RecordId),
    /// From `VariableType::PermutationTable`.
    PermTable,
    /// From `VariableType::InterpolatedNoiseSampler`.
    Sampler,
}

/// Floats are stored as their bits (`to_bits()`), so equality is exact and NaN payloads are kept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lit {
    Bool(bool),
    I8(i8),
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
}

/// Grid dimensions. Unvalidated: `>= 1` is a checker rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dims {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Three `f64`s as bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct F64x3 {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ambient {
    Origin,
    Pos3,
    OriginScale,
    PositionScale,
}

// ---------------------------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------------------------

/// The same variants as `tungsten_ir::BinaryOperator`, in the same order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    ShiftLeft,
    ShiftRight,
    LogicalShiftRight,
    BitOr,
    BitAnd,
    BitXor,
}

/// `UnaryOperator::Ref(_)` is the identity and is removed during lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnOp {
    Negate,
    Not,
}

// ---------------------------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum Expr {
    Lit(Lit),
    Local(LocalId),
    Const(ConstId),
    Host(HostId),
    Table(TableId),
    Ambient(Ambient),
    Density { input: InputId, index: Option<Box<Expr>> },
    Call { func: FnId, args: Vec<Expr> },
    /// `params` and `ret` are the call site's annotation; the checker compares them with the decl.
    ExternCall { ext: ExternId, params: Vec<Type>, ret: Type, args: Vec<Expr> },
    Binary { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    Unary { op: UnOp, operand: Box<Expr> },
    /// `ty` is the annotation (`type_of_field`).
    Field { base: Box<Expr>, sel: FieldSel, ty: Type },
    Index { array: Box<Expr>, index: Box<Expr> },
    MakeVec3 { x: Box<Expr>, y: Box<Expr>, z: Box<Expr> },
    MakePos3 { x: Box<Expr>, y: Box<Expr>, z: Box<Expr> },
    /// Fields in written order. Exactly-once is a checker rule.
    MakeRecord { record: RecordId, fields: Vec<FieldInit> },
    ArrayLit(Vec<Expr>),
    Cast { to: Type, operand: Box<Expr> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldSel {
    Vec3(Axis),
    Pos3(Axis),
    /// Record id and field index.
    Record(RecordId, usize),
}

#[derive(Debug, PartialEq, Eq)]
pub struct FieldInit {
    pub field: usize,
    pub value: Expr,
}

// ---------------------------------------------------------------------------------------------
// Statements and functions
// ---------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum Stmt {
    Assign { target: LocalId, value: Expr },
    Return(Expr),
    If { cond: Expr, then_branch: Vec<Stmt>, else_branch: Vec<Stmt> },
    While { cond: Expr, body: Vec<Stmt> },
    Repeat { count: usize, body: Vec<Stmt> },
    Break,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    /// `locals[0..params]` are the parameters, in order.
    pub params: usize,
    pub locals: Vec<Type>,
    pub ret: Type,
    /// `Function.host_inputs`.
    pub hosts: Vec<HostId>,
    pub body: Vec<Stmt>,
}

// ---------------------------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub main: FnId,
    /// The unit's density inputs, in its arena order.
    pub inputs: Vec<InputId>,
    pub tables: Vec<TableId>,
    pub hosts: Vec<HostId>,
    pub helpers: Vec<FnId>,
    pub consts: Vec<ConstId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputDecl {
    /// The unit whose `density_inputs` arena holds this input.
    pub owner: UnitId,
    /// `DensityInput.density_function`.
    pub source: UnitId,
    pub dims: Dims,
    pub origin_scale: F64x3,
    pub position_scale: F64x3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub unit: UnitId,
    pub dims: Dims,
    pub origin_scale: F64x3,
    pub position_scale: F64x3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostDecl {
    pub ty: Type,
}

/// `PerlinNoise` → `Perm`, `Base3DNoise` → `Sampler`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableKind {
    Perm,
    Sampler,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableDecl {
    pub kind: TableKind,
}

/// No declared type: the checker derives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstDecl {
    pub init: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub units: Vec<Unit>,
    pub functions: Vec<Function>,
    pub consts: Vec<ConstDecl>,
    pub hosts: Vec<HostDecl>,
    pub tables: Vec<TableDecl>,
    pub inputs: Vec<InputDecl>,
    pub entries: Vec<Entry>,
    pub externs: Vec<ExternDecl>,
    pub records: Vec<RecordDecl>,
}

} // verus!

// `Clone` for the recursive types is written outside `verus!{}`: the pinned Verus rejects a
// derived `Clone` on a type that contains itself (through `Box`/`Vec`) as a possibly
// non-terminating cycle. These are plain exec impls with no spec; nothing in the spec relies on
// them.

impl Clone for Type {
    fn clone(&self) -> Self {
        match self {
            Type::Bool => Type::Bool,
            Type::I8 => Type::I8,
            Type::I32 => Type::I32,
            Type::I64 => Type::I64,
            Type::F32 => Type::F32,
            Type::F64 => Type::F64,
            Type::Vec3 => Type::Vec3,
            Type::Pos3 => Type::Pos3,
            Type::Array(t, n) => Type::Array(t.clone(), *n),
            Type::Record(r) => Type::Record(*r),
            Type::PermTable => Type::PermTable,
            Type::Sampler => Type::Sampler,
        }
    }
}

impl Clone for Expr {
    fn clone(&self) -> Self {
        match self {
            Expr::Lit(l) => Expr::Lit(l.clone()),
            Expr::Local(l) => Expr::Local(*l),
            Expr::Const(c) => Expr::Const(*c),
            Expr::Host(h) => Expr::Host(*h),
            Expr::Table(t) => Expr::Table(*t),
            Expr::Ambient(a) => Expr::Ambient(a.clone()),
            Expr::Density { input, index } => Expr::Density { input: *input, index: index.clone() },
            Expr::Call { func, args } => Expr::Call { func: *func, args: args.clone() },
            Expr::ExternCall { ext, params, ret, args } => Expr::ExternCall {
                ext: *ext,
                params: params.clone(),
                ret: ret.clone(),
                args: args.clone(),
            },
            Expr::Binary { op, left, right } => Expr::Binary {
                op: op.clone(),
                left: left.clone(),
                right: right.clone(),
            },
            Expr::Unary { op, operand } => Expr::Unary { op: op.clone(), operand: operand.clone() },
            Expr::Field { base, sel, ty } => Expr::Field {
                base: base.clone(),
                sel: sel.clone(),
                ty: ty.clone(),
            },
            Expr::Index { array, index } => Expr::Index { array: array.clone(), index: index.clone() },
            Expr::MakeVec3 { x, y, z } => Expr::MakeVec3 { x: x.clone(), y: y.clone(), z: z.clone() },
            Expr::MakePos3 { x, y, z } => Expr::MakePos3 { x: x.clone(), y: y.clone(), z: z.clone() },
            Expr::MakeRecord { record, fields } => Expr::MakeRecord { record: *record, fields: fields.clone() },
            Expr::ArrayLit(es) => Expr::ArrayLit(es.clone()),
            Expr::Cast { to, operand } => Expr::Cast { to: to.clone(), operand: operand.clone() },
        }
    }
}

impl Clone for FieldInit {
    fn clone(&self) -> Self {
        FieldInit { field: self.field, value: self.value.clone() }
    }
}

impl Clone for Stmt {
    fn clone(&self) -> Self {
        match self {
            Stmt::Assign { target, value } => Stmt::Assign { target: *target, value: value.clone() },
            Stmt::Return(e) => Stmt::Return(e.clone()),
            Stmt::If { cond, then_branch, else_branch } => Stmt::If {
                cond: cond.clone(),
                then_branch: then_branch.clone(),
                else_branch: else_branch.clone(),
            },
            Stmt::While { cond, body } => Stmt::While { cond: cond.clone(), body: body.clone() },
            Stmt::Repeat { count, body } => Stmt::Repeat { count: *count, body: body.clone() },
            Stmt::Break => Stmt::Break,
        }
    }
}
