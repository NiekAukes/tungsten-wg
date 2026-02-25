/*
SPMT (Single Program Multi Target) is a model for representing density functions in a way that can be easily evaluated
on both CPU and GPU.
The idea is to have a single representation of the density function that can be compiled to both CPU and GPU code,
without having to write separate code for each target.

*/

use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    rc::Rc,
};

pub struct SPMT<'m> {
    pub density_functions: Vec<DensityFunctionRef<'m>>,
    pub functions: Vec<FunctionRef<'m>>,
    pub main_density_functions: Vec<DensityFunctionRef<'m>>,
}

/// A density function is a function that takes in x, y, z coordinates and returns a density value.
/// its signature is at least f(position: Vec3) -> f32
/// However, it can have other inputs, such as inputs of other density functions
#[derive(Debug, Clone)]
pub struct DensityFunction<'m> {
    pub canonical_name: Option<String>,
    pub density_inputs: Vec<DensityInput<'m>>,
    pub body: Vec<Statement<'m>>,
    pub variables: Vec<Rc<Variable>>,
    pub helper_functions: Vec<FunctionRef<'m>>,
}

#[derive(Debug, Clone)]
pub struct Function<'m> {
    pub canonical_name: Option<String>,
    pub parameters: Vec<Rc<Variable>>,
    pub body: Vec<Statement<'m>>,
    pub variables: Vec<Rc<Variable>>,
}

pub type DensityFunctionRef<'m> = Interned<'m, DensityFunction<'m>>;
pub type FunctionRef<'m> = Interned<'m, Function<'m>>;

#[derive(PartialEq, Debug, Clone)]
pub struct DensityInput<'m> {
    pub var: Rc<Variable>,
    pub density_function: DensityFunctionRef<'m>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: Option<String>,
    pub t: VariableType,
}

#[derive(Clone, PartialEq)]
pub enum VariableType {
    DensityInput,
    Vec3,
    F32,
    I32,
}

impl Debug for VariableType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariableType::DensityInput => write!(f, "density"),
            VariableType::Vec3 => write!(f, "vec3"),
            VariableType::F32 => write!(f, "f32"),
            VariableType::I32 => write!(f, "i32"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement<'m> {
    Assign {
        target: Rc<Variable>,
        value: Expression<'m>,
    },
    Return(Expression<'m>),
    If {
        condition: Expression<'m>,
        then_branch: Vec<Statement<'m>>,
        else_branch: Vec<Statement<'m>>,
    },
    While {
        condition: Expression<'m>,
        body: Vec<Statement<'m>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression<'m> {
    Variable(Rc<Variable>),
    Literal(f64),
    FunctionCall {
        function: FunctionRef<'m>,
        parameters: Vec<Expression<'m>>,
    },
    ExternCall {
        function_name: String,
        parameters: Vec<Expression<'m>>,
    },
    DensityVariable(DensityInput<'m>),

    DensityFunctionCall {
        function: DensityFunctionRef<'m>,
        position: Box<Expression<'m>>,
    },

    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression<'m>>,
        right: Box<Expression<'m>>,
    },

    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression<'m>>,
    },

    Field {
        base: Box<Expression<'m>>,
        field: String,
    },

    MakeVec3 {
        x: Box<Expression<'m>>,
        y: Box<Expression<'m>>,
        z: Box<Expression<'m>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOperator {
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOperator {
    Negate,
}

pub struct Interned<'m, T: ?Sized>(&'m T);

impl<'m, T> PartialEq for Interned<'m, T> {
    fn eq(&self, other: &Self) -> bool {
        // pointer equality is sufficient since we intern all functions
        std::ptr::eq(self.0, other.0)
    }
}

impl<'m, T> Eq for Interned<'m, T> {}

impl<'m, T> Hash for Interned<'m, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // pointer hash is sufficient since we intern all functions
        std::ptr::hash(self.0, state);
    }
}

impl<'m, T> std::ops::Deref for Interned<'m, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'m, T> Debug for Interned<'m, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Interned({:p})", self.0)
    }
}

impl<'m, T> Clone for Interned<'m, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'m, T> Copy for Interned<'m, T> {}

impl<'m, T> Interned<'m, T> {
    pub fn new(value: &'m T) -> Self {
        Interned(value)
    }
}

impl<'m> Function<'m> {
    pub fn add_statement(&mut self, statement: Statement<'m>) {
        self.body.push(statement);
    }

    pub fn add_variable(&mut self, variable: Rc<Variable>) {
        self.variables.push(variable);
    }
}

impl<'m> DensityFunction<'m> {
    pub fn add_statement(&mut self, statement: Statement<'m>) {
        self.body.push(statement);
    }
    pub fn add_variable(&mut self, variable: Rc<Variable>) {
        self.variables.push(variable);
    }
}

pub trait Addr {
    fn addr(&self) -> *const ();
}

impl<T> Addr for &T {
    fn addr(&self) -> *const () {
        *self as *const T as *const ()
    }
}

impl<T> Addr for std::rc::Rc<T> {
    fn addr(&self) -> *const () {
        self.as_ref() as *const T as *const ()
    }
}

impl<T> Addr for Interned<'_, T> {
    fn addr(&self) -> *const () {
        self.0 as *const T as *const ()
    }
}
