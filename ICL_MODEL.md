# ICL (Intermediate C/Rust Language) Model

ICL is an intermediate representation language for low-level CPU code that bridges SPMT (density functions) and library generation. It provides a flexible, typed intermediate representation suitable for modeling CPU-level operations.

## Overview

ICL captures the essential elements of low-level imperative programming languages:

- **Functions** with typed parameters and return types
- **Variables** with type information and mutability semantics
- **Expressions** with operators and function calls
- **Statements** for control flow (if/else, loops)
- **Types** including primitives, pointers, structs, and arrays

## Core Components

### Types

```rust
pub enum Type {
    // Primitives
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F32, F64,
    Bool,
    Void,

    // Composite
    Pointer(Box<Type>),
    Struct(String),
    Array(Box<Type>, usize),
}
```

### Variables

Variables are identified by name and type with mutability semantics:

```rust
pub struct Variable {
    pub name: Option<String>,
    pub t: Type,
    pub mutable: bool,
}
```

### Functions

Functions are the primary organizational unit:

```rust
pub struct Function<'m> {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
    pub body: Vec<Statement<'m>>,
    pub variables: Vec<Rc<Variable>>,
    pub inline: bool,
}
```

### Statements

Supported control flow structures:

```rust
pub enum Statement<'m> {
    Assign { target: Rc<Variable>, value: Expression<'m> },
    Return(Option<Expression<'m>>),
    If { condition: Expression<'m>, then_branch: Vec<Statement<'m>>, else_branch: Option<Vec<Statement<'m>>> },
    While { condition: Expression<'m>, body: Vec<Statement<'m>> },
    Declare { variable: Rc<Variable>, init: Option<Expression<'m>> },
    FunctionCall { function: FunctionRef<'m>, arguments: Vec<Expression<'m>> },
    Block(Vec<Statement<'m>>),
}
```

### Expressions

Expressions produce typed values:

```rust
pub enum Expression<'m> {
    Variable(Rc<Variable>),
    IntLiteral(i64),
    FloatLiteral(f64),
    BoolLiteral(bool),
    BinaryOp { op: BinaryOperator, left: Box<Expression<'m>>, right: Box<Expression<'m>> },
    UnaryOp { op: UnaryOperator, operand: Box<Expression<'m>> },
    FunctionCall { function: FunctionRef<'m>, arguments: Vec<Expression<'m>> },
    ExternCall { function_name: String, argument_types: Vec<Type>, return_type: Type, arguments: Vec<Expression<'m>> },
    Field { base: Box<Expression<'m>>, field: String },
    Index { base: Box<Expression<'m>>, index: Box<Expression<'m>> },
    Deref(Box<Expression<'m>>),
    Ref(Box<Expression<'m>>),
    Cast { expr: Box<Expression<'m>>, to_type: Type },
    StructInit { struct_name: String, fields: Vec<(String, Expression<'m>)> },
}
```

## Usage Examples

### Creating a Simple Function

```rust
use cpu_lang::{model::*, codegen};
use std::rc::Rc;

// Create function: add(a: i32, b: i32) -> i32
let mut add_func = Function::new("add".to_string(), Type::I32);
add_func.add_parameter("a".to_string(), Type::I32);
add_func.add_parameter("b".to_string(), Type::I32);

// Create variables
let a_var = Rc::new(Variable {
    name: Some("a".to_string()),
    t: Type::I32,
    mutable: false,
});
let b_var = Rc::new(Variable {
    name: Some("b".to_string()),
    t: Type::I32,
    mutable: false,
});

// Create expression: a + b
let add_expr = Expression::BinaryOp {
    op: BinaryOperator::Add,
    left: Box::new(Expression::Variable(a_var)),
    right: Box::new(Expression::Variable(b_var)),
};

// Add return statement
add_func.add_statement(Statement::Return(Some(add_expr)));

// Generate Rust code
let rust_code = codegen::generate_rust_function(&Interned::new(&add_func));
println!("{}", rust_code);
```

### Function with Control Flow

```rust
// Create function: max(a: i32, b: i32) -> i32
let mut max_func = Function::new("max".to_string(), Type::I32);
max_func.add_parameter("a".to_string(), Type::I32);
max_func.add_parameter("b".to_string(), Type::I32);

// Create condition: a > b
let condition = Expression::BinaryOp {
    op: BinaryOperator::Greater,
    left: Box::new(Expression::Variable(a_var.clone())),
    right: Box::new(Expression::Variable(b_var.clone())),
};

// Add if statement
max_func.add_statement(Statement::If {
    condition,
    then_branch: vec![Statement::Return(Some(Expression::Variable(a_var)))],
    else_branch: Some(vec![Statement::Return(Some(Expression::Variable(b_var)))]),
});
```

### Creating Structs

```rust
// Create a Vec3 struct
let mut vec3 = Struct::new("Vec3".to_string());
vec3.add_field("x".to_string(), Type::F32);
vec3.add_field("y".to_string(), Type::F32);
vec3.add_field("z".to_string(), Type::F32);

// Use in function
let mut func = Function::new("vec3_length".to_string(), Type::F32);
func.add_parameter("v".to_string(), Type::Struct("Vec3".to_string()));
```

## Conversion from SPMT

ICL can be generated from SPMT density functions using the conversion module:

```rust
use cpu_lang::convert;
use spmt::model as spmt;

// Convert SPMT density function to ICL function
let (icl_func, converter) = convert::spmt_density_function_to_icl(&spmt_df);

// Generate Rust code
let rust_code = codegen::generate_rust_function(&Interned::new(&icl_func));
```

The conversion automatically:

- Maps SPMT types to appropriate ICL types
- Converts SPMT statements and expressions
- Adds coordinate parameters (x, y, z) for density functions
- Handles density inputs as function parameters

## Code Generation

The `codegen` module generates production-ready Rust code from ICL:

```rust
use cpu_lang::codegen;

// Generate a single function
let rust_code = codegen::generate_rust_function(&func_ref);

// Generate entire module with all functions and structs
let module_code = codegen::generate_rust_module(&icl);
```

Generated features:

- Proper Rust syntax with correct indentation
- Type annotations
- Inline hints for optimization
- Proper operator precedence
- Memory safety annotations (references, dereferencing)

## Operators

### Binary Operators

- Arithmetic: `Add`, `Subtract`, `Multiply`, `Divide`, `Modulo`
- Comparison: `Equal`, `NotEqual`, `Less`, `LessEqual`, `Greater`, `GreaterEqual`
- Logical: `And`, `Or`
- Bitwise: `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `LeftShift`, `RightShift`

### Unary Operators

- Arithmetic: `Negate`
- Logical: `Not`
- Bitwise: `BitwiseNot`

## Interning Pattern

Functions and structs use the `Interned` pointer wrapper for efficient deduplication and comparison:

```rust
pub struct Interned<'m, T: ?Sized>(&'m T);
```

This enables:

- Pointer equality for fast function comparison
- Automatic deduplication in allocators
- Use as HashMap/HashSet keys

## Lifetime Management

ICL uses Rust lifetimes to tie function and variable references to their context:

```rust
pub struct Function<'m> { ... }
pub struct ICL<'m> { ... }
```

This ensures:

- Safe reference tracking
- Memory coherence in the intermediate representation
- Proper cleanup and deallocation

## Testing

Run the included examples:

```bash
cargo test cpu_lang -- --nocapture
```

Tests demonstrate:

- Creating simple functions
- Building control flow structures
- Defining structs
- Code generation
- SPMT conversion

## Future Extensions

Potential enhancements:

- Loop unrolling and optimization passes
- SIMD vector types
- Function inlining
- Dead code elimination
- Type inference system
- Pattern matching for expressions
- Generic function templates
