use naga::proc::index;

use crate::{
    parse::model::{Density, Spline, SplinePoint, SplineValue},
    spmt::model::{BinaryOperator, Expression, Function, Statement, Var, Variable, VariableType},
    transform_spmt::{density::DensityBuilder, newvar, prefixvar},
};

impl<'a, 'm> DensityBuilder<'a, 'm> {
    pub fn lower_spline_definition_new(
        &mut self,
        spline: Spline<'a>,
        canonical_name: Option<String>,
    ) -> Expression<'m> {
        // create spline cache
        let mut cache: Vec<(Spline<'a>, Expression<'m>)> = Vec::new();
        let expr = self.lower_spline_with_cache(spline, &mut cache);
        return expr;
    }

    fn lower_spline_with_cache(
        &mut self,
        spline: Spline<'a>,
        cache: &mut Vec<(Spline<'a>, Expression<'m>)>,
    ) -> Expression<'m> {
        // check if spline already exists in cache
        for (cached_spline, cached_expr) in cache.iter() {
            if self.is_equal_spline(&spline, cached_spline) {
                return cached_expr.clone();
            }
        }

        // lower the spline and add it to the cache
        let expr = self.lower_spline_new(spline.clone(), cache);
        cache.push((spline, expr.clone()));
        expr
    }

    fn lower_spline_new(
        &mut self,
        spline: Spline<'a>,
        cache: &mut Vec<(Spline<'a>, Expression<'m>)>,
    ) -> Expression<'m> {
        let mut coordinates = Vec::new();
        let mut derivatives = Vec::new();
        //coordinates.push(-1_000_000.0_f32);
        let mut values = Vec::new();
        for point in spline.spline_points.iter() {
            coordinates.push(point.location as f32);
            derivatives.push(point.derivative as f32);
            // if derivatives.len() == 1 {
            //     derivatives.push(point.derivative as f32);
            // }
            match point.value {
                SplineValue::Const(c) => values.push(Expression::Float(c as f32)),
                SplineValue::Spline(s) => {
                    let expr = self.lower_spline_with_cache(s, cache);
                    values.push(expr);
                }
            }
        }

        // add a final coordinate and derivative for extrapolation
        // coordinates.push(1_000_000.0_f32);
        // derivatives.push(derivatives[derivatives.len() - 1]);
        // values

        // create the coordinate array as constant and value array as variable
        let coordinate_array_var = prefixvar(
            self.arena,
            "spline_coordinates",
            VariableType::Array(Box::new(VariableType::F32), coordinates.len()),
        );
        let coordinate_array_expr =
            Expression::ArrayLiteral(coordinates.iter().map(|&c| Expression::Float(c)).collect());
        self.add_constant(coordinate_array_var, coordinate_array_expr);

        let derivative_array_var = prefixvar(
            self.arena,
            "spline_derivatives",
            VariableType::Array(Box::new(VariableType::F32), derivatives.len()),
        );
        let derivative_array_expr =
            Expression::ArrayLiteral(derivatives.iter().map(|&d| Expression::Float(d)).collect());
        self.add_constant(derivative_array_var, derivative_array_expr);

        let value_array_var = prefixvar(
            self.arena,
            "spline_values",
            VariableType::Array(Box::new(VariableType::F32), values.len()),
        );
        let value_array_expr = Expression::ArrayLiteral(values.clone());
        self.add_variable(value_array_var);
        self.add_statement(Statement::Assign {
            target: value_array_var.clone(),
            value: value_array_expr,
        });

        let input_var = prefixvar(self.arena, "coordinate", VariableType::F32);
        self.add_variable(input_var);
        // lower the coordinate into a function that takes the input coordinate and returns the value
        let coordinate = self.lower_density_input(spline.coordinate, None, None);
        let coordinate_expr = Expression::DensityVariable(coordinate, None);
        self.add_statement(Statement::Assign {
            target: input_var,
            value: coordinate_expr.clone(),
        });

        // create the binary search expression
        let binary_search_expr = self.make_binary_search(
            Expression::Variable(coordinate_array_var),
            coordinates.len(),
            Expression::Variable(input_var),
        );

        let index_var = prefixvar(self.arena, "spline_index", VariableType::I32);
        self.add_variable(index_var);
        self.add_statement(Statement::Assign {
            target: index_var.clone(),
            value: binary_search_expr,
        });

        return self.make_hermite(
            Expression::Variable(coordinate_array_var),
            Expression::Variable(value_array_var),
            Expression::Variable(derivative_array_var),
            Expression::Variable(input_var),
            Expression::Variable(index_var),
        );

        //make hermite interpolation expression
        // return self.make_hermite(
        //     Expression::ArrayAccess {
        //         array: Box::new(Expression::Variable(derivative_array_var)),
        //         index: Box::new(Expression::Variable(index_minus_one_var)),
        //     },
        //     Expression::ArrayAccess {
        //         array: Box::new(Expression::Variable(derivative_array_var)),
        //         index: Box::new(Expression::Variable(index_var)),
        //     },
        //     Expression::ArrayAccess {
        //         array: Box::new(Expression::Variable(coordinate_array_var)),
        //         index: Box::new(Expression::Variable(index_minus_one_var)),
        //     },
        //     Expression::ArrayAccess {
        //         array: Box::new(Expression::Variable(coordinate_array_var)),
        //         index: Box::new(Expression::Variable(index_var)),
        //     },
        //     Expression::ArrayAccess {
        //         array: Box::new(Expression::Variable(value_array_var)),
        //         index: Box::new(Expression::Variable(index_minus_one_var)),
        //     },
        //     Expression::ArrayAccess {
        //         array: Box::new(Expression::Variable(value_array_var)),
        //         index: Box::new(Expression::Variable(index_var)),
        //     },
        // );
    }

    fn make_binary_search(
        &self,
        array_expr: Expression<'m>,
        n: usize,
        target_expr: Expression<'m>,
    ) -> Expression<'m> {
        // make a binary search tree for the array of size n
        // return the index of the first element that is greater than or equal to the target
        Expression::ExternCall {
            function_name: "binary_search".to_string(),
            parameters: vec![array_expr, target_expr],
            parameter_types: vec![
                VariableType::Array(Box::new(VariableType::F32), n),
                VariableType::F32,
            ],
        }
    }

    fn make_hermite(
        &mut self,
        spline_locations_var: Expression<'m>,
        spline_values_var: Expression<'m>,
        spline_derivatives_var: Expression<'m>,
        coordinate: Expression<'m>,
        index: Expression<'m>,
    ) -> Expression<'m> {
        /*

        #[inline(always)]
        pub fn advanced_hermite<const N: usize>(
            spline_locations: [f32; N],
            spline_values: [f32; N],
            spline_derivatives: [f32; N],
            coordinate: f32,
            index: i32,
        ) -> f32 {
            // let h_minus_g = second_value - first_value;
            // hermite(
            //     t,
            //     first_value,
            //     second_value,
            //     first_derivative,
            //     second_derivative,
            //     h_minus_g,
            // )
            let index: usize = index as usize;
            if index == 0 || index >= N {
                // extrapolate
                let value = spline_values[index.min(N)];
                let derivative = spline_derivatives[index.min(N)];
                let location = spline_locations[index.min(N)];
                return value + derivative * (coordinate - location);
            }
            let index_minus_1 = index - 1;
            let h_minus_g = spline_values[index] - spline_values[index_minus_1];
            let t = (coordinate - spline_locations[index_minus_1])
                / (spline_locations[index] - spline_locations[index_minus_1]);
            let value_minus_1 = spline_values[index_minus_1];
            let value = spline_values[index];
            let derivative_minus_1 = spline_derivatives[index_minus_1];
            let derivative = spline_derivatives[index];
            hermite(
                t,
                value_minus_1,
                value,
                derivative_minus_1,
                derivative,
                h_minus_g,
            )
        } */

        Expression::ExternCall {
            function_name: "advanced_hermite".to_string(),
            parameters: vec![
                spline_locations_var,
                spline_values_var,
                spline_derivatives_var,
                coordinate,
                index,
            ],
            parameter_types: vec![
                VariableType::Array(Box::new(VariableType::F32), 0), // We can use 0 here since the function will be extern and won't actually use the size
                VariableType::Array(Box::new(VariableType::F32), 0),
                VariableType::Array(Box::new(VariableType::F32), 0),
                VariableType::F32,
                VariableType::I32,
            ],
        }
    }
}

trait SplineLowering<'a, 'm> {
    fn is_1d_spline(&self, spline: &Spline<'a>) -> bool;
    fn is_equal_spline(&self, spline1: &Spline<'a>, spline2: &Spline<'a>) -> bool;
}

impl<'a, 'm> SplineLowering<'a, 'm> for DensityBuilder<'a, 'm> {
    fn is_1d_spline(&self, spline: &Spline<'a>) -> bool {
        // check if all points are const
        for point in spline.spline_points.iter() {
            match point.value {
                SplineValue::Const(_) => continue,
                SplineValue::Spline(_) => return false,
            }
        }
        true
    }

    fn is_equal_spline(&self, spline1: &Spline<'a>, spline2: &Spline<'a>) -> bool {
        if spline1.spline_points.len() != spline2.spline_points.len() {
            return false;
        }
        for (point1, point2) in spline1
            .spline_points
            .iter()
            .zip(spline2.spline_points.iter())
        {
            if point1.derivative != point2.derivative
                || point1.location != point2.location
                || point1.value != point2.value
            {
                return false;
            }
        }
        true
    }
}
