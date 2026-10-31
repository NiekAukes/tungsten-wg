//! The core IR: the form of a tungsten program that the verified checker, the spec semantics and
//! the proofs work on. See `README.md`.
//!
//! Everything here sits inside `verus!{}` and depends only on `vstd`. A plain `cargo build`
//! erases the ghost code, so using this crate never needs Verus. Verification runs through
//! `scripts/verify.sh`.
//!
//! The IR holds plain data: dense `usize` ids, resolved types, no names, floats as bits. Names
//! live in `tungsten-check`'s side tables. [`wf`] defines index-bounds well-formedness as
//! `spec fn`s.

#![allow(unused_imports)]
// The derived `Clone`s have no spec; nothing in the spec uses them.
#![cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]

use vstd::prelude::*;

pub mod ir;
pub mod registry;
pub mod wf;

pub use ir::*;
pub use registry::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_and_compare() {
        let p = Program {
            units: vec![Unit {
                main: FnId(0),
                inputs: vec![],
                tables: vec![TableId(0)],
                hosts: vec![HostId(0)],
                helpers: vec![],
                consts: vec![ConstId(0)],
            }],
            functions: vec![Function {
                params: 1,
                locals: vec![Type::F64, Type::Array(Box::new(Type::Record(RecordId(0))), 2)],
                ret: Type::F64,
                hosts: vec![HostId(0)],
                body: vec![
                    Stmt::If {
                        cond: Expr::Binary {
                            op: BinOp::Less,
                            left: Box::new(Expr::Local(LocalId(0))),
                            right: Box::new(Expr::Lit(Lit::F64(f64::NAN.to_bits()))),
                        },
                        then_branch: vec![Stmt::Break],
                        else_branch: vec![],
                    },
                    Stmt::Return(Expr::Field {
                        base: Box::new(Expr::Ambient(Ambient::Origin)),
                        sel: FieldSel::Vec3(Axis::X),
                        ty: Type::F64,
                    }),
                ],
            }],
            consts: vec![ConstDecl { init: Expr::Lit(Lit::F32((-0.0f32).to_bits())) }],
            hosts: vec![HostDecl { ty: Type::I64 }],
            tables: vec![TableDecl { kind: TableKind::Perm }],
            inputs: vec![],
            entries: vec![Entry {
                unit: UnitId(0),
                dims: Dims { x: 1, y: 1, z: 1 },
                origin_scale: F64x3 { x: 1.0f64.to_bits(), y: 1.0f64.to_bits(), z: 1.0f64.to_bits() },
                position_scale: F64x3 { x: 0, y: 0, z: 0 },
            }],
            externs: vec![ExternDecl { params: vec![Type::F64], ret: Type::F64, domain_error: false }],
            records: vec![RecordDecl { fields: vec![Type::F64, Type::I32] }],
        };
        let q = p.clone();
        assert_eq!(p, q);
        assert_eq!(format!("{p:?}"), format!("{q:?}"));

        let mut r = p.clone();
        r.consts[0].init = Expr::Lit(Lit::F32(0.0f32.to_bits()));
        assert_ne!(p, r, "-0.0 and 0.0 are different literals");
        assert_eq!(LocalId(3), LocalId(3));
        assert!(LocalId(2) < LocalId(3));
    }
}
