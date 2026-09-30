use crate::spmt::builder::Builder;
use crate::spmt::model::*;

macro_rules! impl_binop {
    ($($trait:ident :: $method:ident => $op:ident),* $(,)?) => {$(
        impl<'m, R: Into<Expression<'m>>> std::ops::$trait<R> for Expression<'m> {
            type Output = Expression<'m>;
            fn $method(self, rhs: R) -> Expression<'m> {
                Expression::BinaryOp {
                    op: BinaryOperator::$op,
                    left: Box::new(self),
                    right: Box::new(rhs.into()),
                }
            }
        }

        impl<'m, R: Into<Expression<'m>>> std::ops::$trait<R> for Var<'m> {
            type Output = Expression<'m>;
            fn $method(self, rhs: R) -> Expression<'m> {
                Expression::BinaryOp {
                    op: BinaryOperator::$op,
                    left: Box::new(self.into()),
                    right: Box::new(rhs.into()),
                }
            }
        }
    )*};
}
impl_binop!(Add::add => Add, Sub::sub => Subtract, Mul::mul => Multiply, Div::div => Divide);

impl<'m> std::ops::Neg for Expression<'m> {
    type Output = Expression<'m>;
    fn neg(self) -> Self::Output {
        Expression::UnaryOp { op: UnaryOperator::Negate, operand: Box::new(self) }
    }
}

impl<'m> std::ops::Neg for Var<'m> {
    type Output = Expression<'m>;
    fn neg(self) -> Self::Output {
        Expression::UnaryOp { op: UnaryOperator::Negate, operand: Box::new(self.into()) }
    }
}

// impl<'m> From<Var<'m>> for Expression<'m> {
//     fn from(v: Var<'m>) -> Self {
//         Expression::Variable(v)
//     }
// }

macro_rules! impl_from_lit {
    ($($ty:ty => $variant:ident),*) => {$(
        impl<'m> From<$ty> for Expression<'m> {
            fn from(v: $ty) -> Self { Expression::$variant(v) }
        }
    )*};
}
impl_from_lit!(f32 => Float, f64 => Double, i32 => Int, i64 => Long);

impl<'m> From<Var<'m>> for Expression<'m> {
    fn from(v: Var<'m>) -> Self { Expression::Variable(v) }
}

macro_rules! impl_cmp {
    ($($name:ident => $op:ident),*) => {
        impl<'m> Expression<'m> {$(
            pub fn $name(self, rhs: impl Into<Expression<'m>>) -> Expression<'m> {
                Expression::BinaryOp {
                    op: BinaryOperator::$op,
                    left: Box::new(self),
                    right: Box::new(rhs.into()),
                }
            }
        )*}

        impl<'m> Var<'m> {$(
            pub fn $name(self, rhs: impl Into<Expression<'m>>) -> Expression<'m> {
                Expression::BinaryOp {
                    op: BinaryOperator::$op,
                    left: Box::new(self.into()),
                    right: Box::new(rhs.into()),
                }
            }
        )*}
    };
}
impl_cmp!(lt => Less, le => LessEqual, gt => Greater, ge => GreaterEqual,
          eq_ => Equal, ne => NotEqual, and => And, or => Or);


macro_rules! extern_fns {
    ($($name:ident($($arg:ident),*) : $ty:ident),* $(,)?) => {$(
        pub fn $name<'m>($($arg: impl Into<Expression<'m>>),*) -> Expression<'m> {
            Expression::ExternCall {
                parameter_types: vec![$( { let _ = &$arg; VariableType::$ty } ),*],
                function_name: stringify!($name).to_string(),
                parameters: vec![$($arg.into()),*],
            }
        }
    )*};
}

extern_fns! {
    sin(x): F32, cos(x): F32, sqrt(x): F32, floor(x): F32,
    pow(a, b): F32, min(a, b): F32, max(a, b): F32, clamp(x, lo, hi): F32,
}

macro_rules! vars {
    ($b:expr; $($name:ident : $t:ident),* $(,)?) => {
        $( let $name = $b.local(stringify!($name), VariableType::$t); )*
    };
}

macro_rules! body {
    (@acc $v:ident;) => {};
    (@acc $v:ident; return $e:expr; $($rest:tt)*) => {
        $v.push(Statement::Return(($e).into()));
        body!(@acc $v; $($rest)*);
    };
    (@acc $v:ident; break; $($rest:tt)*) => {
        $v.push(Statement::Break);
        body!(@acc $v; $($rest)*);
    };
    (@acc $v:ident; repeat ($n:expr) { $($b:tt)* } $($rest:tt)*) => {
        $v.push(Statement::Repeat { count: $n, body: body!($($b)*) });
        body!(@acc $v; $($rest)*);
    };
    (@acc $v:ident; while ($c:expr) { $($b:tt)* } $($rest:tt)*) => {
        $v.push(Statement::While { condition: ($c).into(), body: body!($($b)*) });
        body!(@acc $v; $($rest)*);
    };
    (@acc $v:ident; if ($c:expr) { $($t:tt)* } else { $($e:tt)* } $($rest:tt)*) => {
        $v.push(Statement::If {
            condition: ($c).into(),
            then_branch: body!($($t)*),
            else_branch: body!($($e)*),
        });
        body!(@acc $v; $($rest)*);
    };
    (@acc $v:ident; if ($c:expr) { $($t:tt)* } $($rest:tt)*) => {
        $v.push(Statement::If {
            condition: ($c).into(),
            then_branch: body!($($t)*),
            else_branch: Vec::new(),
        });
        body!(@acc $v; $($rest)*);
    };
    (@acc $v:ident; $t:ident = $e:expr; $($rest:tt)*) => {
        $v.push(Statement::Assign { target: $t, value: ($e).into() });
        body!(@acc $v; $($rest)*);
    };
    ($($t:tt)*) => {{
        let mut v = Vec::new();
        body!(@acc v; $($t)*);
        v
    }};
}

macro_rules! vec3 {
    ($x:expr, $y:expr, $z:expr) => {
        Expression::Construct {
            t: VariableType::Vec3,
            args: vec![($x).into(), ($y).into(), ($z).into()],
        }
    };
}

macro_rules! pos3 {
    ($x:expr, $y:expr, $z:expr) => {
        Expression::Construct {
            t: VariableType::Pos3,
            args: vec![($x).into(), ($y).into(), ($z).into()],
        }
    };
}

macro_rules! array {
    ($($e:expr),* $(,)?) => { Expression::ArrayLiteral(vec![$(($e).into()),*]) };
}

fn test() {
    let arena = bumpalo::Bump::new();
    let mut b = Builder::new(&arena);
    vars!(b; x: F32, y: F32, acc: F32);

    let body = body! {
        acc = 0.0f32;
        repeat (4) {
            acc = acc + sin(x * 2.0f32) * y;
        }
        if (acc.gt(1.0f32)) {
            return 1.0f32;
        } else {
            return -acc;
        }
    };
}