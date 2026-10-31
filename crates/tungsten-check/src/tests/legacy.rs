//! Programs from `tungsten-legacy`'s builders, after `convert`, lower against a registry without
//! errors. (The program is the one `tungsten-parse/tests/legacy.rs` round-trips.)

use bumpalo::Bump;
use tungsten_legacy::builder::{Builder, SPMTBuilder};
use tungsten_legacy::conversion::convert;
use tungsten_legacy::model::*;

use super::registry;

fn unit<'m>(
    arena: &'m Bump,
    name: &str,
    body: impl FnOnce(&mut Builder<'m>) -> Vec<Statement<'m>>,
) -> DensityFunctionRef<'m> {
    let mut b = Builder::new(arena).named(name).returns(VariableType::F64);
    let stmts = body(&mut b);
    b.set_body(stmts);
    b.finish_compute_unit()
}

fn build(arena: &Bump) -> SPMT<'_> {
    let mut h = Builder::new(arena)
        .named("minecraft:accumulate")
        .returns(VariableType::F64);
    let t = h.param("t", VariableType::F64);
    let acc = h.local("acc", VariableType::F64);
    h.set_body(vec![
        Statement::Assign {
            target: acc,
            value: 0.0f64.into(),
        },
        Statement::Repeat {
            count: 3,
            body: vec![
                Statement::Assign {
                    target: acc,
                    value: acc + t,
                },
                Statement::If {
                    condition: acc.gt(10.0f64),
                    then_branch: vec![Statement::Break],
                    else_branch: vec![],
                },
            ],
        },
        Statement::Return(Expression::Variable(acc)),
    ]);
    let helper = h.finish_function();

    let mut s = Builder::new(arena)
        .named("minecraft:seed")
        .returns(VariableType::F64);
    let seed = s.alloc_var(Name::Named("seed".into()), VariableType::I64);
    s.set_body(vec![Statement::Return(Expression::ExplicitCast {
        to: VariableType::F64,
        expr: Box::new(Expression::Variable(seed)),
    })]);
    let mut seed_unit = s.build_compute_unit();
    seed_unit.host_inputs = vec![HostInput::Static(seed)];
    let seed_unit: DensityFunctionRef = Interned::new(arena.alloc(seed_unit));

    let noise = unit(arena, "minecraft:overworld/noise", |n| {
        let base = n.constant("base", VariableType::F64, 0.25f64);
        let scale = n.constant(
            "scale",
            VariableType::F64,
            Expression::Variable(base) * 2.0f64,
        );
        let perm = Expression::PermutationTable(PermutationTableInput::PerlinNoise {
            ident: "minecraft:temperature".into(),
            subident: Some("first".into()),
            subident_index: 0,
        });
        vec![Statement::Return(
            Expression::ExternCall {
                function_name: "sample".into(),
                parameters: vec![perm, 1.0f64.into()],
                parameter_types: vec![VariableType::PermutationTable, VariableType::F64],
                return_type: VariableType::F64,
            } * Expression::Variable(scale),
        )]
    });
    let shared = unit(arena, "minecraft:shared", |b| {
        let seed_in = b.density(seed_unit, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1, 1, 1));
        vec![Statement::Return(seed_in.get() + 1.0f64)]
    });
    let top = unit(arena, "minecraft:overworld/final_density", |b| {
        let s = b.density(shared, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (4, 4, 4));
        let n = b.density(noise, (1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (2, 2, 2));
        let x = b.local("x", VariableType::F64);
        vec![
            Statement::Assign {
                target: x,
                value: s.get() + n.at(0i32),
            },
            Statement::If {
                condition: Expression::Variable(x).gt(0.0f64),
                then_branch: vec![Statement::Assign {
                    target: x,
                    value: helper.call([Expression::Variable(x)]),
                }],
                else_branch: vec![],
            },
            Statement::Return(Expression::Variable(x) * s.get()),
        ]
    });
    SPMT {
        density_functions: vec![seed_unit, noise, shared, top],
        functions: vec![helper],
        main_density_functions: vec![MainDensityFunction {
            density_function: top,
            dimensions: (4, 4, 4),
            scaled_origin: (1.0, 1.0, 1.0),
            scaled_position: (1.0, 1.0, 1.0),
        }],
    }
}

#[test]
fn converted_program_lowers() {
    let arena = Bump::new();
    let p = convert(&build(&arena)).expect("conversion");
    let l = super::ok(&p, &registry());
    assert_eq!(l.map.externs, vec!["sin".to_string(), "sample".to_string()]);
    assert_eq!(l.program.entries.len(), 1);
    assert_eq!(l.program.inputs.len(), 3);
}
