// transform_spmt/spline.rs

use std::rc::Rc;

use crate::transform_spmt::density::DensityBuilder;
use crate::transform_spmt::{anonvar, newvar};
use crate::{
    parse::model::{Spline, SplinePoint, SplineValue},
    spmt::model::{
        BinaryOperator, Expression, Function, FunctionRef, Statement, Variable, VariableType,
    },
};
use bumpalo::Bump;

impl<'a, 'm> DensityBuilder<'a, 'm> {
    pub fn lower_spline_definition(
        &mut self,
        spline: Spline<'a>,
        canonical_name: Option<String>,
    ) -> FunctionRef<'m> {
        let mut function: Function<'m> = Function {
            canonical_name: canonical_name,
            parameters: Vec::new(),
            body: Vec::new(),
            variables: Vec::new(),
        };
        let p: Rc<Variable> = Rc::new(Variable {
            name: self.p.name.clone(),
            t: self.p.t.clone(),
        });
        function.parameters.push(p.clone());

        // Compute coordinate
        let coordinate = newvar("coordinate", VariableType::F32);
        function.variables.push(coordinate.clone());

        let old_function = self.switch_function(function);

        let coord_expr = self.lower_density(spline.coordinate);

        self.add_statement(Statement::Assign {
            target: coordinate.clone(),
            value: coord_expr,
        });

        // -----------------------------------------
        // Sort points
        // -----------------------------------------
        let mut points = Vec::from(spline.spline_points);
        points.sort_by(|a, b| a.location.partial_cmp(&b.location).unwrap());

        // -----------------------------------------
        // Build interpolation chain
        // -----------------------------------------
        let r = self.build_spline_chain(&points, &p, &coordinate);
        let function: FunctionRef<'m> = self.finish_and_continue(r, old_function);
        function
    }

    fn build_spline_chain(
        &mut self,
        points: &[SplinePoint<'a>],
        p: &std::rc::Rc<Variable>,
        input: &std::rc::Rc<Variable>,
    ) -> Expression<'m> {
        // Base case: one point → extrapolation
        if points.len() == 1 {
            return self.make_extrapolation(&points[0], p, input);
        }

        let first = &points[0];
        let second = &points[1];

        // if (input < second.location)
        let cond = Expression::BinaryOp {
            op: BinaryOperator::Less,
            left: Box::new(Expression::Variable(input.clone())),
            right: Box::new(Expression::Float(second.location)),
        };

        let mut then_branch = Vec::new();
        then_branch.push(self.make_hermite_return(first, second, p, input));

        self.add_statement(Statement::If {
            condition: cond,
            then_branch,
            else_branch: Vec::new(),
        });

        return self.build_spline_chain(&points[1..], p, input);
    }

    fn lower_spline_value_expr(
        &mut self,
        value: &SplineValue<'a>,
        p: &std::rc::Rc<Variable>,
    ) -> Expression<'m> {
        match value {
            SplineValue::Const(c) => Expression::Float(*c),

            SplineValue::Spline(def) => {
                // recursive spline -> generate nested function call
                //let fname = format!("spline_nested");
                let fref = self.lower_spline_definition(def, None);

                Expression::FunctionCall {
                    function: fref,
                    parameters: vec![Expression::Variable(p.clone())],
                }
            }
        }
    }

    fn make_extrapolation(
        &mut self,
        point: &SplinePoint<'a>,
        p: &std::rc::Rc<Variable>,
        input: &std::rc::Rc<Variable>,
    ) -> Expression<'m> {
        Expression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                left: Box::new(Expression::BinaryOp {
                    op: BinaryOperator::Subtract,
                    left: Box::new(Expression::Variable(input.clone())),
                    right: Box::new(Expression::Float(point.location)),
                }),
                right: Box::new(Expression::Float(point.derivative)),
            }),
            right: Box::new(self.lower_spline_value_expr(&point.value, p)),
        }
    }

    fn make_hermite_return(
        &mut self,
        first: &SplinePoint<'a>,
        second: &SplinePoint<'a>,
        p: &std::rc::Rc<Variable>,
        input: &std::rc::Rc<Variable>,
    ) -> Statement<'m> {
        let t = Expression::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expression::BinaryOp {
                op: BinaryOperator::Subtract,
                left: Box::new(Expression::Variable(input.clone())),
                right: Box::new(Expression::Float(first.location)),
            }),
            right: Box::new(Expression::BinaryOp {
                op: BinaryOperator::Subtract,
                left: Box::new(Expression::Float(second.location)),
                right: Box::new(Expression::Float(first.location)),
            }),
        };

        let f = self.lower_spline_value_expr(&first.value, p);
        let s = self.lower_spline_value_expr(&second.value, p);

        Statement::Return(Expression::ExternCall {
            function_name: "hermite".into(),
            parameters: vec![
                t,
                f,
                s,
                Expression::Float(first.derivative),
                Expression::Float(second.derivative),
            ],
            parameter_types: vec![
                VariableType::F32,
                VariableType::F32,
                VariableType::F32,
                VariableType::F32,
                VariableType::F32,
            ],
        })
    }
}
