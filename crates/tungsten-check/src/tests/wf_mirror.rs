//! A plain-Rust mirror of `tungsten_core::wf`, clause by clause. Kept in sync with `wf.rs` by
//! review; replaced by the checker's verified exec code later.

use tungsten_core::*;

fn wf_type(p: &Program, t: &Type) -> bool {
    match t {
        Type::Array(inner, _) => wf_type(p, inner),
        Type::Record(r) => r.0 < p.records.len(),
        _ => true,
    }
}

fn wf_types(p: &Program, ts: &[Type]) -> bool {
    ts.iter().all(|t| wf_type(p, t))
}

fn wf_sel(p: &Program, sel: &FieldSel) -> bool {
    match sel {
        FieldSel::Record(r, i) => r.0 < p.records.len() && *i < p.records[r.0].fields.len(),
        _ => true,
    }
}

fn wf_expr(p: &Program, nlocals: usize, e: &Expr) -> bool {
    let w = |e: &Expr| wf_expr(p, nlocals, e);
    match e {
        Expr::Lit(_) | Expr::Ambient(_) => true,
        Expr::Local(l) => l.0 < nlocals,
        Expr::Const(c) => c.0 < p.consts.len(),
        Expr::Host(h) => h.0 < p.hosts.len(),
        Expr::Table(t) => t.0 < p.tables.len(),
        Expr::Density { input, index } => {
            input.0 < p.inputs.len() && index.as_deref().is_none_or(w)
        }
        Expr::Call { func, args } => func.0 < p.functions.len() && args.iter().all(w),
        Expr::ExternCall {
            ext,
            params,
            ret,
            args,
        } => {
            ext.0 < p.externs.len()
                && wf_types(p, params)
                && wf_type(p, ret)
                && args.iter().all(w)
        }
        Expr::Binary { left, right, .. } => w(left) && w(right),
        Expr::Unary { operand, .. } => w(operand),
        Expr::Field { base, sel, ty } => w(base) && wf_sel(p, sel) && wf_type(p, ty),
        Expr::Index { array, index } => w(array) && w(index),
        Expr::MakeVec3 { x, y, z } | Expr::MakePos3 { x, y, z } => w(x) && w(y) && w(z),
        Expr::MakeRecord { record, fields } => {
            record.0 < p.records.len()
                && fields
                    .iter()
                    .all(|f| f.field < p.records[record.0].fields.len() && w(&f.value))
        }
        Expr::ArrayLit(elems) => elems.iter().all(w),
        Expr::Cast { to, operand } => wf_type(p, to) && w(operand),
    }
}

fn wf_stmt(p: &Program, nlocals: usize, s: &Stmt) -> bool {
    let all = |b: &[Stmt]| b.iter().all(|s| wf_stmt(p, nlocals, s));
    match s {
        Stmt::Assign { target, value } => target.0 < nlocals && wf_expr(p, nlocals, value),
        Stmt::Return(e) => wf_expr(p, nlocals, e),
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => wf_expr(p, nlocals, cond) && all(then_branch) && all(else_branch),
        Stmt::While { cond, body } => wf_expr(p, nlocals, cond) && all(body),
        Stmt::Repeat { body, .. } => all(body),
        Stmt::Break => true,
    }
}

fn wf_function(p: &Program, f: &Function) -> bool {
    f.params <= f.locals.len()
        && wf_types(p, &f.locals)
        && wf_type(p, &f.ret)
        && f.hosts.iter().all(|h| h.0 < p.hosts.len())
        && f.body.iter().all(|s| wf_stmt(p, f.locals.len(), s))
}

fn wf_unit(p: &Program, u: &Unit) -> bool {
    u.main.0 < p.functions.len()
        && u.inputs.iter().all(|i| i.0 < p.inputs.len())
        && u.tables.iter().all(|t| t.0 < p.tables.len())
        && u.hosts.iter().all(|h| h.0 < p.hosts.len())
        && u.helpers.iter().all(|f| f.0 < p.functions.len())
        && u.consts.iter().all(|c| c.0 < p.consts.len())
}

fn wf_input_links(p: &Program) -> bool {
    p.inputs.iter().enumerate().all(|(i, d)| {
        p.units
            .get(d.owner.0)
            .is_some_and(|u| u.inputs.contains(&InputId(i)))
    }) && p.units.iter().enumerate().all(|(u, unit)| {
        unit.inputs
            .iter()
            .all(|i| p.inputs.get(i.0).is_some_and(|d| d.owner == UnitId(u)))
    })
}

/// Panics, naming the first failing clause, unless `wf_program(p)`.
pub fn assert_wf(p: &Program) {
    for (i, u) in p.units.iter().enumerate() {
        assert!(wf_unit(p, u), "unit {i} is not wf: {u:?}");
    }
    for (i, f) in p.functions.iter().enumerate() {
        assert!(wf_function(p, f), "function {i} is not wf: {f:?}");
    }
    for (i, c) in p.consts.iter().enumerate() {
        assert!(wf_expr(p, 0, &c.init), "const {i} is not wf: {c:?}");
    }
    for (i, h) in p.hosts.iter().enumerate() {
        assert!(wf_type(p, &h.ty), "host {i} is not wf");
    }
    for (i, d) in p.inputs.iter().enumerate() {
        assert!(
            d.owner.0 < p.units.len() && d.source.0 < p.units.len(),
            "input {i} is not wf"
        );
    }
    for (i, e) in p.entries.iter().enumerate() {
        assert!(e.unit.0 < p.units.len(), "entry {i} is not wf");
    }
    for (i, d) in p.externs.iter().enumerate() {
        assert!(
            wf_types(p, &d.params) && wf_type(p, &d.ret),
            "extern {i} is not wf"
        );
    }
    for (i, d) in p.records.iter().enumerate() {
        assert!(wf_types(p, &d.fields), "record {i} is not wf");
    }
    assert!(wf_input_links(p), "input ownership links are not wf");
}

#[test]
fn mirror_rejects_out_of_range_ids() {
    let p = Program {
        units: vec![],
        functions: vec![Function {
            params: 0,
            locals: vec![],
            ret: Type::F64,
            hosts: vec![],
            body: vec![Stmt::Return(Expr::Local(LocalId(0)))],
        }],
        consts: vec![],
        hosts: vec![],
        tables: vec![],
        inputs: vec![],
        entries: vec![],
        externs: vec![],
        records: vec![],
    };
    assert!(!wf_function(&p, &p.functions[0]));
    assert!(!wf_expr(&p, 0, &Expr::Field {
        base: Box::new(Expr::Lit(Lit::Bool(true))),
        sel: FieldSel::Record(RecordId(0), 0),
        ty: Type::F64,
    }));
    assert!(!wf_type(&p, &Type::Array(Box::new(Type::Record(RecordId(3))), 1)));
}
