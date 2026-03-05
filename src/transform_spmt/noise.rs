use std::rc::Rc;

use crate::{
    parse::model::NormalNoise,
    spmt::model::{
        BinaryOperator, DensityFunction, Expression, Function, Statement, Variable, VariableType,
    },
    transform_spmt::newvar,
};

pub fn lower_normal_noise<'m>(noise: NormalNoise, cname: Option<String>) -> Function<'m> {
    let mut variables = Vec::new();
    let mut body = Vec::new();

    /* -----------------------------
    Input position
    ----------------------------- */
    let p = newvar("p", VariableType::Vec3);

    /* -----------------------------
    freq0 = pow(2.0, firstOctave)
    ----------------------------- */
    let freq0 = newvar("freq0", VariableType::F32);

    body.push(Statement::Assign {
        target: freq0.clone(),
        value: Expression::ExternCall {
            function_name: "pow".into(),
            parameters: vec![
                Expression::Float(2.0),
                Expression::Float(noise.first_octave as f64),
            ],
        },
    });

    variables.push(freq0.clone());

    let mut freqs = vec![freq0];

    /* -----------------------------
    freq[i] = freq[i-1] * 2.0
    ----------------------------- */
    for i in 1..noise.amplitudes.len() {
        let prev = freqs.last().unwrap();

        let freq = newvar(&format!("freq{}", i), VariableType::F32);
        body.push(Statement::Assign {
            target: freq.clone(),
            value: Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                left: Box::new(Expression::Variable(prev.clone())),
                right: Box::new(Expression::Float(2.0)),
            },
        });

        variables.push(freq.clone());
        freqs.push(freq);
    }

    /* -----------------------------
    n[i] = perlin(p * freq[i]) * amp[i]
    ----------------------------- */
    let mut noise_terms = Vec::new();

    for (i, amp) in noise.amplitudes.iter().enumerate() {
        let n = newvar(&format!("n{}", i), VariableType::F32);

        let scaled_pos = Expression::BinaryOp {
            op: BinaryOperator::Multiply,
            left: Box::new(Expression::Variable(p.clone())),
            right: Box::new(Expression::Variable(freqs[i].clone())),
        };

        let perlin = Expression::ExternCall {
            function_name: "perlin".into(),
            parameters: vec![scaled_pos],
        };

        let scaled_noise = Expression::BinaryOp {
            op: BinaryOperator::Multiply,
            left: Box::new(perlin),
            right: Box::new(Expression::Float(*amp)),
        };

        body.push(Statement::Assign {
            target: n.clone(),
            value: scaled_noise,
        });

        variables.push(n.clone());
        noise_terms.push(Expression::Variable(n.clone()));
    }

    /* -----------------------------
    return n0 + n1 + ...
    ----------------------------- */
    let sum = noise_terms
        .into_iter()
        .reduce(|a, b| Expression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(a),
            right: Box::new(b),
        })
        .unwrap();

    body.push(Statement::Return(sum));

    Function {
        canonical_name: None,
        //density_inputs: vec![],
        //helper_functions: vec![],
        parameters: vec![p],
        body,
        variables,
    }
}
