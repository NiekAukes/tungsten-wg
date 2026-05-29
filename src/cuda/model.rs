/*
CUDA C++ model for code generation.

This module defines an intermediate representation for CUDA C++ code.
It captures:
- Kernel functions (__global__) and device functions (__device__)
- CUDA-specific types: vector types (float2, float4, int2, int4), pointers
- CUDA built-ins: threadIdx, blockIdx, blockDim, gridDim
- Memory qualifiers: __shared__, __constant__, __device__
- Control flow: if/else, while, C-style for
- Expressions and operators

The model mirrors the structure of the RCL (CPU) model so that SPMT → CUDA
transformations are straightforward.
*/

use std::{fmt::Debug, hash::Hash, rc::Rc};

use crate::spmt::model::{Interned, Name};

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

/// Root structure for a CUDA translation unit.
#[derive(Debug, Clone)]
pub struct CudaModule<'m> {
    /// Verbatim `#include` / `#define` lines placed at the top of the file.
    pub includes: Vec<String>,
    /// File-scope variable declarations (e.g. `__constant__` arrays).
    pub global_vars: Vec<GlobalVar<'m>>,
    /// Struct / C++ class definitions.
    pub structs: Vec<StructRef<'m>>,
    /// Non-kernel helper functions (`__device__` or `__host__ __device__`).
    pub device_functions: Vec<FunctionRef<'m>>,
    /// Kernel entry-points (`__global__`).
    pub kernels: Vec<FunctionRef<'m>>,
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Supported types in CUDA C++.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Type {
    // --- C primitive types ----------------------------------------------
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float,
    Double,
    Bool,
    Void,

    // --- CUDA vector types (subset) -------------------------------------
    Float2,
    Float4,
    Int2,
    Int4,

    // --- Composite types ------------------------------------------------
    /// Raw (non-const) pointer.
    Pointer(Box<Type>),
    /// `const T*`
    ConstPointer(Box<Type>),
    /// Fixed-size C array: `T name[N]`.
    Array(Box<Type>, usize),
    /// Named struct / typedef.
    Struct(String),
}

impl Type {
    pub fn is_signed_int(&self) -> bool {
        matches!(self, Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64)
    }

    pub fn is_unsigned_int(&self) -> bool {
        matches!(
            self,
            Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64
        )
    }

    pub fn is_int(&self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::Float | Type::Double)
    }

    pub fn is_vector(&self) -> bool {
        matches!(self, Type::Float2 | Type::Float4 | Type::Int2 | Type::Int4)
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int8 => write!(f, "int8_t"),
            Type::Int16 => write!(f, "int16_t"),
            Type::Int32 => write!(f, "int32_t"),
            Type::Int64 => write!(f, "int64_t"),
            Type::UInt8 => write!(f, "uint8_t"),
            Type::UInt16 => write!(f, "uint16_t"),
            Type::UInt32 => write!(f, "uint32_t"),
            Type::UInt64 => write!(f, "uint64_t"),
            Type::Float => write!(f, "float"),
            Type::Double => write!(f, "double"),
            Type::Bool => write!(f, "bool"),
            Type::Void => write!(f, "void"),
            Type::Float2 => write!(f, "float2"),
            Type::Float4 => write!(f, "float4"),
            Type::Int2 => write!(f, "int2"),
            Type::Int4 => write!(f, "int4"),
            Type::Pointer(t) => write!(f, "{}*", t),
            Type::ConstPointer(t) => write!(f, "const {}*", t),
            Type::Array(t, size) => write!(f, "{}[{}]", t, size),
            Type::Struct(name) => write!(f, "{}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory & function qualifiers
// ---------------------------------------------------------------------------

/// CUDA memory-space qualifier for variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryQualifier {
    /// `__shared__`
    Shared,
    /// `__constant__`
    Constant,
    /// `__device__`
    Device,
}

impl std::fmt::Display for MemoryQualifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryQualifier::Shared => write!(f, "__shared__"),
            MemoryQualifier::Constant => write!(f, "__constant__"),
            MemoryQualifier::Device => write!(f, "__device__"),
        }
    }
}

/// Execution-space qualifier for functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionQualifier {
    /// `__global__` — kernel called from host, runs on device.
    Global,
    /// `__device__` — called from device only.
    Device,
    /// `__host__` — called from host only (default in CUDA C++).
    Host,
    /// `__host__ __device__` — callable from both.
    HostDevice,
}

impl std::fmt::Display for FunctionQualifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionQualifier::Global => write!(f, "__global__"),
            FunctionQualifier::Device => write!(f, "__device__"),
            FunctionQualifier::Host => write!(f, "__host__"),
            FunctionQualifier::HostDevice => write!(f, "__host__ __device__"),
        }
    }
}

// ---------------------------------------------------------------------------
// Struct definition
// ---------------------------------------------------------------------------

/// A C struct (or typedef struct) definition.
#[derive(Debug, Clone)]
pub struct CudaStruct {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

pub type StructRef<'m> = Interned<'m, CudaStruct>;

impl CudaStruct {
    pub fn new(name: String) -> Self {
        CudaStruct {
            name,
            fields: Vec::new(),
        }
    }

    pub fn add_field(&mut self, name: String, t: Type) {
        self.fields.push((name, t));
    }
}

// ---------------------------------------------------------------------------
// Global variable
// ---------------------------------------------------------------------------

/// A file-scope variable declaration, optionally with a memory qualifier
/// and an initializer expression.
#[derive(Debug, Clone)]
pub struct GlobalVar<'m> {
    pub name: String,
    pub t: Type,
    pub qualifier: Option<MemoryQualifier>,
    pub init: Option<Expression<'m>>,
}

// ---------------------------------------------------------------------------
// Function
// ---------------------------------------------------------------------------

/// A parameter to a CUDA function.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub t: Type,
    /// Whether the parameter is declared `const`.
    pub is_const: bool,
}

/// A local variable used inside a function body.
#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: Name,
    pub t: Type,
    pub memory_qualifier: Option<MemoryQualifier>,
}

/// A CUDA function definition (kernel or device helper).
#[derive(Debug, Clone)]
pub struct CudaFunction<'m> {
    pub qualifier: FunctionQualifier,
    pub name: Option<String>,
    /// Optional C++ template parameters, e.g. `["typename T", "int N"]`.
    pub template_params: Vec<String>,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
    pub body: Vec<Statement<'m>>,
    pub variables: Vec<Rc<Variable>>,
    pub is_inline: bool,
    pub is_extern_c: bool,
}

pub type FunctionRef<'m> = Interned<'m, CudaFunction<'m>>;

impl<'m> CudaFunction<'m> {
    pub fn new(qualifier: FunctionQualifier, name: Option<String>, return_type: Type) -> Self {
        CudaFunction {
            qualifier,
            name,
            template_params: Vec::new(),
            parameters: Vec::new(),
            return_type,
            body: Vec::new(),
            variables: Vec::new(),
            is_inline: false,
            is_extern_c: false,
        }
    }

    pub fn add_parameter(&mut self, name: String, t: Type, is_const: bool) {
        self.parameters.push(Parameter { name, t, is_const });
    }

    pub fn add_statement(&mut self, statement: Statement<'m>) {
        self.body.push(statement);
    }

    pub fn add_variable(&mut self, variable: Rc<Variable>) {
        self.variables.push(variable);
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// Statements that form the body of a CUDA function.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement<'m> {
    /// `target = value;`
    Assign {
        target: Rc<Variable>,
        value: Expression<'m>,
    },

    /// `target[index] = value;`
    ArrayAssign {
        target: Rc<Variable>,
        index: Expression<'m>,
        value: Expression<'m>,
    },

    /// `return expr;` or `return;`
    Return(Option<Expression<'m>>),

    /// `if (condition) { ... } else { ... }`
    If {
        condition: Expression<'m>,
        then_branch: Vec<Statement<'m>>,
        else_branch: Option<Vec<Statement<'m>>>,
    },

    /// `while (condition) { ... }`
    While {
        condition: Expression<'m>,
        body: Vec<Statement<'m>>,
    },

    /// C-style `for (init; condition; increment) { body }`
    For {
        init: Option<Box<Statement<'m>>>,
        condition: Option<Expression<'m>>,
        increment: Option<Box<Statement<'m>>>,
        body: Vec<Statement<'m>>,
    },

    /// Variable declaration with optional initializer.
    Declare {
        variable: Rc<Variable>,
        init: Option<Expression<'m>>,
        is_const: bool,
    },

    /// An expression used as a statement (e.g. a function call).
    ExprStatement(Expression<'m>),

    /// A scoped block `{ ... }`.
    Block(Vec<Statement<'m>>),

    /// `__syncthreads();`
    SyncThreads,

    /// `break;`
    Break,

    /// Verbatim CUDA C++ code inserted directly into the output.
    InlineCuda(String),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// Axis selector for CUDA built-in vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Axis::X => write!(f, "x"),
            Axis::Y => write!(f, "y"),
            Axis::Z => write!(f, "z"),
        }
    }
}

/// Expressions that produce values in CUDA C++.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression<'m> {
    // --- Variables & literals -------------------------------------------
    Variable(Rc<Variable>),
    I32Literal(i32),
    U32Literal(u32),
    I64Literal(i64),
    U64Literal(u64),
    F32Literal(f32),
    F64Literal(f64),
    BoolLiteral(bool),

    // --- Arithmetic / logical / bitwise ---------------------------------
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression<'m>>,
        right: Box<Expression<'m>>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression<'m>>,
    },

    // --- Function calls -------------------------------------------------
    /// Call to a function whose definition is in this module (interned ref).
    FunctionCall {
        function: FunctionRef<'m>,
        arguments: Vec<Expression<'m>>,
    },
    /// Call to a function known only by name at this point (device builtins,
    /// math intrinsics, externally-defined helpers, ...).
    LateBoundCall {
        function_name: String,
        arguments: Vec<Expression<'m>>,
    },

    // --- Compound access ------------------------------------------------
    /// `base.field`
    Field {
        base: Box<Expression<'m>>,
        field: String,
    },
    /// `base[index]`
    Index {
        base: Box<Expression<'m>>,
        index: Box<Expression<'m>>,
    },
    /// `*expr`
    Deref(Box<Expression<'m>>),
    /// `&expr`
    AddressOf(Box<Expression<'m>>),
    /// `(T)expr` — C-style cast.
    Cast {
        expr: Box<Expression<'m>>,
        to_type: Type,
    },

    // --- Initialization -------------------------------------------------
    /// `StructName { field: value, ... }` (aggregate initializer).
    StructInit {
        struct_name: String,
        fields: Vec<(String, Expression<'m>)>,
    },

    // --- CUDA built-ins -------------------------------------------------
    /// `threadIdx.x / threadIdx.y / threadIdx.z`
    ThreadIdx(Axis),
    /// `blockIdx.x / blockIdx.y / blockIdx.z`
    BlockIdx(Axis),
    /// `blockDim.x / blockDim.y / blockDim.z`
    BlockDim(Axis),
    /// `gridDim.x / gridDim.y / gridDim.z`
    GridDim(Axis),

    // --- Aggregate initializers -----------------------------------------
    /// Aggregate array initializer: `{e1, e2, e3}`
    ArrayLiteral(Vec<Expression<'m>>),

    // --- Escape hatch ---------------------------------------------------
    /// Verbatim CUDA C++ expression inserted directly into the output.
    InlineCuda(String),
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    // Comparison
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    // Logical
    And,
    Or,
    // Bitwise
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Negate,
    Not,
    BitwiseNot,
}

// ---------------------------------------------------------------------------
// CudaModule impl
// ---------------------------------------------------------------------------

impl<'m> CudaModule<'m> {
    pub fn new() -> Self {
        CudaModule {
            includes: Vec::new(),
            global_vars: Vec::new(),
            structs: Vec::new(),
            device_functions: Vec::new(),
            kernels: Vec::new(),
        }
    }

    pub fn add_include(&mut self, include: String) {
        self.includes.push(include);
    }

    pub fn add_struct(&mut self, s: StructRef<'m>) {
        self.structs.push(s);
    }

    pub fn add_device_function(&mut self, f: FunctionRef<'m>) {
        self.device_functions.push(f);
    }

    pub fn add_kernel(&mut self, k: FunctionRef<'m>) {
        self.kernels.push(k);
    }

    pub fn add_global_var(&mut self, v: GlobalVar<'m>) {
        self.global_vars.push(v);
    }
}
