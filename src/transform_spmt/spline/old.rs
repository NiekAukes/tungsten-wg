use crate::{
    parse::model::{Spline, SplinePoint, SplineValue},
    spmt::model::{BinaryOperator, Expression, Function, Statement, Var, Variable, VariableType},
    transform_spmt::{density::DensityBuilder, newvar},
};

/// The old spline implementation, which is the verified correct implementation. It does not work on the GPU however.
/// This is used to verify the new spline implementation, which is faster and works on the GPU.
///

impl<'a, 'm> DensityBuilder<'a, 'm> {
    /// Old spline implementation: builds a chain of if-statements for interpolation
    pub fn lower_spline_definition_old(
        &mut self,
        spline: Spline<'a>,
        canonical_name: Option<String>,
    ) -> Expression<'m> {
        let mut function: Function<'m> = Function {
            canonical_name: canonical_name,
            parameters: Vec::new(),
            body: Vec::new(),
            variables: Vec::new(),
            return_type: VariableType::F32,
        };
        let p: Var<'m> = Var::new(self.arena.alloc(Variable {
            name: self.p.name.clone(),
            t: self.p.t.clone(),
        }));
        function.parameters.push(p.clone());

        // Compute coordinate
        let coordinate = newvar(self.arena, "coordinate", VariableType::F32);
        function.variables.push(coordinate.clone());

        let old_function = self.switch_function(function);

        let coord_input = self.lower_density_input(spline.coordinate, None, None);
        let coord_expr = Expression::DensityVariable(coord_input, None);

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

        let r = self.build_spline_chain(&points, p, coordinate);
        let function_ref = self.finish_and_continue(r, old_function);

        // Wrap the function call in an expression
        Expression::FunctionCall {
            function: function_ref,
            parameters: vec![Expression::Variable(self.p.clone())],
        }
    }

    fn build_spline_chain(
        &mut self,
        points: &[SplinePoint<'a>],
        p: Var<'m>,
        input: Var<'m>,
    ) -> Expression<'m> {
        // build the initial extrapolation
        // when input < first.location, extrapolate using the first point's derivative
        let first = &points[0];
        let initial_extrapolation = self.make_extrapolation(first, p.clone(), input.clone());

        // wrap it in an if statement that checks if input < first.location
        let cond = Expression::BinaryOp {
            op: BinaryOperator::Less,
            left: Box::new(Expression::Variable(input.clone())),
            right: Box::new(Expression::Float(first.location as f32)),
        };
        let mut then_branch = Vec::new();
        then_branch.push(Statement::Return(initial_extrapolation));
        self.add_statement(Statement::If {
            condition: cond,
            then_branch,
            else_branch: Vec::new(),
        });

        self.continue_spline_chain(points, p, input)
    }

    fn continue_spline_chain(
        &mut self,
        points: &[SplinePoint<'a>],
        p: Var<'m>,
        input: Var<'m>,
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
            right: Box::new(Expression::Float(second.location as f32)),
        };

        let mut then_branch = Vec::new();
        then_branch.push(self.make_hermite_return(first, second, p, input));

        self.add_statement(Statement::If {
            condition: cond,
            then_branch,
            else_branch: Vec::new(),
        });

        return self.continue_spline_chain(&points[1..], p, input);
    }

    fn lower_spline_value_expr(&mut self, value: &SplineValue<'a>, p: Var<'m>) -> Expression<'m> {
        match value {
            SplineValue::Const(c) => Expression::Float(*c as f32),

            SplineValue::Spline(def) => {
                // recursive spline -> generate nested function call
                let spline_expr = self.lower_spline_definition(*def, None);
                // If it's a function call, we can use it directly; otherwise wrap it
                spline_expr
            }
        }
    }

    fn make_extrapolation(
        &mut self,
        point: &SplinePoint<'a>,
        p: Var<'m>,
        input: Var<'m>,
    ) -> Expression<'m> {
        Expression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                left: Box::new(Expression::BinaryOp {
                    op: BinaryOperator::Subtract,
                    left: Box::new(Expression::Variable(input.clone())),
                    right: Box::new(Expression::Float(point.location as f32)),
                }),
                right: Box::new(Expression::Float(point.derivative as f32)),
            }),
            right: Box::new(self.lower_spline_value_expr(&point.value, p)),
        }
    }

    fn make_hermite_return(
        &mut self,
        first: &SplinePoint<'a>,
        second: &SplinePoint<'a>,
        p: Var<'m>,
        input: Var<'m>,
    ) -> Statement<'m> {
        /*
        let h_minus_g = -0.15_f32 - -0.19_f32;
        let t = (coordinate - -0.19_f32) / h_minus_g;

        return hermite(
            t,
            3.95_f32,
            minecraft_factor_test_function_129124268986176(pos3, /* ... */) as f32,
            0.0_f32,
            0.0_f32,
            h_minus_g
        );
         */
        let h_minus_g = second.location as f32 - first.location as f32;
        let h_minus_g_expr = Expression::Float(h_minus_g);
        let t = Expression::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expression::BinaryOp {
                op: BinaryOperator::Subtract,
                left: Box::new(Expression::Variable(input.clone())),
                right: Box::new(Expression::Float(first.location as f32)),
            }),
            right: Box::new(Expression::BinaryOp {
                op: BinaryOperator::Subtract,
                left: Box::new(Expression::Float(second.location as f32)),
                right: Box::new(Expression::Float(first.location as f32)),
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
                Expression::Float(first.derivative as f32),
                Expression::Float(second.derivative as f32),
                h_minus_g_expr,
            ],
            parameter_types: vec![
                VariableType::F32,
                VariableType::F32,
                VariableType::F32,
                VariableType::F32,
                VariableType::F32,
                VariableType::F32,
            ],
        })
    }
}
