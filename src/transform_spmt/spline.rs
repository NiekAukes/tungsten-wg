// transform_spmt/spline.rs

use serde::de::value;

use crate::parse::model::Density;
use crate::spmt::model::DensityInput;
use crate::transform_spmt::density::{DensityBuilder, make_rpos3};
use crate::transform_spmt::{newvar, prefixvar};
use crate::{
    parse::model::{Spline, SplinePoint, SplineValue},
    spmt::model::{
        BinaryOperator, Expression, Function, FunctionRef, Statement, Var, Variable, VariableType,
    },
};

// impl<'a, 'm> DensityBuilder<'a, 'm> {
//     pub fn lower_spline_definition(
//         &mut self,
//         spline: Spline<'a>,
//         canonical_name: Option<String>,
//     ) -> FunctionRef<'m> {
//         let mut function: Function<'m> = Function {
//             canonical_name: canonical_name,
//             parameters: Vec::new(),
//             body: Vec::new(),
//             variables: Vec::new(),
//         };
//         let p: Var<'m> = Var::new(self.arena.alloc(Variable {
//             name: self.p.name.clone(),
//             t: self.p.t.clone(),
//         }));
//         function.parameters.push(p.clone());

//         // Compute coordinate
//         let coordinate = newvar(self.arena, "coordinate", VariableType::F32);
//         function.variables.push(coordinate.clone());

//         let old_function = self.switch_function(function);

//         let coord_input = self.lower_density_input(spline.coordinate, None);

//         let coord_expr = Expression::DensityVariable(coord_input, None);

//         self.add_statement(Statement::Assign {
//             target: coordinate.clone(),
//             value: coord_expr,
//         });

//         // -----------------------------------------
//         // Sort points
//         // -----------------------------------------
//         let mut points = Vec::from(spline.spline_points);
//         points.sort_by(|a, b| a.location.partial_cmp(&b.location).unwrap());

//         // -----------------------------------------
//         // Build interpolation chain
//         // -----------------------------------------

//         let r = self.build_spline_chain(&points, p, coordinate);
//         let function: FunctionRef<'m> = self.finish_and_continue(r, old_function);
//         function
//     }

//     fn build_spline_chain(
//         &mut self,
//         points: &[SplinePoint<'a>],
//         p: Var<'m>,
//         input: Var<'m>,
//     ) -> Expression<'m> {
//         // build the initial extrapolation
//         // when input < first.location, extrapolate using the first point's derivative
//         let first = &points[0];
//         let initial_extrapolation = self.make_extrapolation(first, p.clone(), input.clone());

//         // wrap it in an if statement that checks if input < first.location
//         let cond = Expression::BinaryOp {
//             op: BinaryOperator::Less,
//             left: Box::new(Expression::Variable(input.clone())),
//             right: Box::new(Expression::Float(first.location)),
//         };
//         let mut then_branch = Vec::new();
//         then_branch.push(Statement::Return(initial_extrapolation));
//         self.add_statement(Statement::If {
//             condition: cond,
//             then_branch,
//             else_branch: Vec::new(),
//         });

//         self.continue_spline_chain(points, p, input)
//     }

//     fn continue_spline_chain(
//         &mut self,
//         points: &[SplinePoint<'a>],
//         p: Var<'m>,
//         input: Var<'m>,
//     ) -> Expression<'m> {
//         // Base case: one point → extrapolation
//         if points.len() == 1 {
//             return self.make_extrapolation(&points[0], p, input);
//         }

//         let first = &points[0];
//         let second = &points[1];

//         // if (input < second.location)
//         let cond = Expression::BinaryOp {
//             op: BinaryOperator::Less,
//             left: Box::new(Expression::Variable(input.clone())),
//             right: Box::new(Expression::Float(second.location)),
//         };

//         let mut then_branch = Vec::new();
//         then_branch.push(self.make_hermite_return(first, second, p, input));

//         self.add_statement(Statement::If {
//             condition: cond,
//             then_branch,
//             else_branch: Vec::new(),
//         });

//         return self.continue_spline_chain(&points[1..], p, input);
//     }

//     fn lower_spline_value_expr(&mut self, value: &SplineValue<'a>, p: Var<'m>) -> Expression<'m> {
//         match value {
//             SplineValue::Const(c) => Expression::Float(*c),

//             SplineValue::Spline(def) => {
//                 // recursive spline -> generate nested function call
//                 //let fname = format!("spline_nested");
//                 let fref = self.lower_spline_definition(def, None);

//                 Expression::FunctionCall {
//                     function: fref,
//                     parameters: vec![Expression::Variable(p.clone())],
//                 }
//             }
//         }
//     }

//     fn make_extrapolation(
//         &mut self,
//         point: &SplinePoint<'a>,
//         p: Var<'m>,
//         input: Var<'m>,
//     ) -> Expression<'m> {
//         Expression::BinaryOp {
//             op: BinaryOperator::Add,
//             left: Box::new(Expression::BinaryOp {
//                 op: BinaryOperator::Multiply,
//                 left: Box::new(Expression::BinaryOp {
//                     op: BinaryOperator::Subtract,
//                     left: Box::new(Expression::Variable(input.clone())),
//                     right: Box::new(Expression::Float(point.location)),
//                 }),
//                 right: Box::new(Expression::Float(point.derivative)),
//             }),
//             right: Box::new(self.lower_spline_value_expr(&point.value, p)),
//         }
//     }

//     fn make_hermite_return(
//         &mut self,
//         first: &SplinePoint<'a>,
//         second: &SplinePoint<'a>,
//         p: Var<'m>,
//         input: Var<'m>,
//     ) -> Statement<'m> {
//         let t = Expression::BinaryOp {
//             op: BinaryOperator::Divide,
//             left: Box::new(Expression::BinaryOp {
//                 op: BinaryOperator::Subtract,
//                 left: Box::new(Expression::Variable(input.clone())),
//                 right: Box::new(Expression::Float(first.location)),
//             }),
//             right: Box::new(Expression::BinaryOp {
//                 op: BinaryOperator::Subtract,
//                 left: Box::new(Expression::Float(second.location)),
//                 right: Box::new(Expression::Float(first.location)),
//             }),
//         };

//         let f = self.lower_spline_value_expr(&first.value, p);
//         let s = self.lower_spline_value_expr(&second.value, p);

//         Statement::Return(Expression::ExternCall {
//             function_name: "hermite".into(),
//             parameters: vec![
//                 t,
//                 f,
//                 s,
//                 Expression::Float(first.derivative),
//                 Expression::Float(second.derivative),
//             ],
//             parameter_types: vec![
//                 VariableType::F32,
//                 VariableType::F32,
//                 VariableType::F32,
//                 VariableType::F32,
//                 VariableType::F32,
//             ],
//         })
//     }
// }

/* New spline idea:

Make a decision tree of the spline points
decision_tree: [(u8, u24, f32)] = [(next_coord_idx, next_decision_idx, location)]
values: [f32] = [value0, value1, value2, ...] (the values of the spline points in the same order as the decision tree)

coord = input[0]
decision_idx = 0
decision = decision_tree[decision_idx]
value = 0.0
prev_value = 0.0
derivative = 0.0
location = 0.0

next_coord_idx, next_decision_idx, location = decision

for _ in max_depth {
    decision_idx = next_decision_idx * (coord >= location) + (coord < location)
    if coord < location && (nc > max_coord_idx || decision_idx > max_decision_idx) {
        value, derivative = values[decision_idx - max_decision_idx]
        prev_value = values[decision_idx - max_decision_idx - 1]
        break
    }

    decision = decision_tree[decision_idx]
    next_coord_idx, next_decision_idx, location = decision
    coord = input[next_coord_idx]
}

// get the value
// interpolate or extrapolate based on the last decision

// extrapolation: return (((coordinate - 1f) * 0.38940096f) + 0.69000006f);
// hermite: fn hermite(t: f32, p0_: f32, p1_: f32, m0_: f32, m1_: f32) -> f32 {
    let t2_ = (t * t);
    let t3_ = (t2_ * t);
    return (((((((2f * t3_) - (3f * t2_)) + 1f) * p0_) + (((t3_ - (2f * t2_)) + t) * m0_)) + (((-2f * t3_) + (3f * t2_)) * p1_)) + ((t3_ - t2_) * m1_));
}

if nc > max_coord_idx {
    return (((coordinate - location) * derivative) + value);
} else {
    return hermite(t, prev_value, value, derivative, next_derivative);
}


*/
struct DecisionTree<'a> {
    decisions: Vec<(Density<'a>, i32, f64)>, // next_coord_idx, next_decision_idx, next_value_idx location
    values: Vec<(f64, f64)>,                 // (value, derivative)
    coordinates: Vec<Density<'a>>,
    max_iterations: usize,
}

impl<'a> DecisionTree<'a> {
    fn new() -> Self {
        DecisionTree {
            decisions: Vec::new(),
            values: Vec::new(),
            coordinates: Vec::new(),
            max_iterations: 0,
        }
    }

    fn finish(self) -> (Vec<(u8, u32, f64)>, Vec<(f64, f64)>, Vec<Density<'a>>) {
        // concretely, we need to convert the decisions to use coordinate indices instead of variables

        let mut inputs = Vec::new();
        let mut decisions = Vec::new();
        // let decisions = self
        //     .decisions
        //     .into_iter()
        //     .map(|(input, next_decision_idx, location)| {
        //         let coord_idx = *coord_indices.get(&input.var).unwrap();
        //         (coord_idx, next_decision_idx, location)
        //     })
        //     .collect();
        let max_decision_idx = self.decisions.len() as i32 - 1;
        for decision in self.decisions.into_iter() {
            let (input, mut next_decision_idx, location) = decision;
            //let coord_idx = *coord_indices.get(&input.var).unwrap();
            let coord_idx = if !inputs.contains(&input) {
                inputs.push(input);
                (inputs.len() - 1) as u8
            } else {
                inputs.iter().position(|i| *i == input).unwrap() as u8
            };

            if next_decision_idx < 0 {
                next_decision_idx = (-next_decision_idx) + max_decision_idx
            }

            decisions.push((coord_idx, next_decision_idx as u32, location));
        }
        (decisions, self.values, inputs)
    }
}

impl<'a, 'm> DensityBuilder<'a, 'm> {
    /// Main entry point for spline lowering - routes to old or new implementation based on settings
    pub fn lower_spline_definition(
        &mut self,
        spline: Spline<'a>,
        canonical_name: Option<String>,
    ) -> Expression<'m> {
        if self.builder_settings.use_new_spline {
            self.lower_spline_definition_new(spline, canonical_name)
        } else {
            self.lower_spline_definition_old(spline, canonical_name)
        }
    }

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

    pub fn lower_spline_definition_new(
        &mut self,
        spline: Spline<'a>,
        canonical_name: Option<String>,
    ) -> Expression<'m> {
        // steps to compute the decision tree and values:
        // 2. build the decision tree by going through the spline
        let tree = self.add_decisions_recursive(&spline);
        let max_iterations = tree.max_iterations;
        let (decisions, values, coordinates) = tree.finish();
        let max_decision_idx = decisions.len() as i32 + 1; // must do +1, otherwise we count double 0
        // 3. generate the function body that traverses the decision tree and returns the correct value

        let decision_tree_var = prefixvar(
            self.arena,
            "decision_tree",
            VariableType::Array(
                Box::new(VariableType::Extern("DecisionTreeNode")), // (next_coord_idx, next_decision_idx, location)
                decisions.len(),
            ),
        );
        let decision_tree_init = self.make_decision_tree_init(&decisions);
        self.add_constant(decision_tree_var.clone(), decision_tree_init);

        let values_var = prefixvar(
            self.arena,
            "values",
            VariableType::Array(Box::new(VariableType::Extern("SplineValue")), values.len()),
        );
        let values_init = self.make_values_init(&values);
        self.add_constant(values_var.clone(), values_init);

        // build the for loop that traverses the decision tree
        let decision_idx_var = prefixvar(self.arena, "decision_idx", VariableType::I32);
        self.add_variable(decision_idx_var.clone());
        self.add_statement(Statement::Assign {
            target: decision_idx_var.clone(),
            value: Expression::Int(1), // start at 1 because 0 is the extrapolation decision
        });

        let value_var = prefixvar(self.arena, "value", VariableType::F32);
        let derivative_var = prefixvar(self.arena, "derivative", VariableType::F32);
        self.add_variable(value_var.clone());
        self.add_variable(derivative_var.clone());

        let mut coordinate_vars = Vec::new();

        for coord in &coordinates {
            let coord_var = prefixvar(self.arena, "coord", VariableType::F32);
            let lowered: Expression<'m> = self.lower_density(coord.clone());
            self.add_statement(Statement::Assign {
                target: coord_var.clone(),
                //value: Expression::DensityVariable(coord.clone(), None),
                value: lowered,
            });
            self.add_variable(coord_var.clone());
            coordinate_vars.push(coord_var);
        }

        let first_coord_var = coordinate_vars[0].clone();
        let coordinate_var = prefixvar(self.arena, "coordinate", VariableType::F32);
        self.add_variable(coordinate_var.clone());
        self.add_statement(Statement::Assign {
            target: coordinate_var.clone(),
            value: Expression::Variable(first_coord_var.clone()),
        });

        let coordinates_array_var = prefixvar(
            self.arena,
            "coordinates",
            VariableType::Array(Box::new(VariableType::F32), coordinates.len()),
        );

        let lits: Vec<Expression<'_>> = coordinate_vars
            .iter()
            .map(|c| Expression::Variable(c.clone()))
            .collect();
        if lits.len() != coordinates.len() {
            println!("Coordinates: {:?}", coordinates.len());
            println!("Lits: {:?}", lits.len());
        }

        self.add_variable(coordinates_array_var.clone());
        self.add_statement(Statement::Assign {
            target: coordinates_array_var.clone(),
            value: Expression::ArrayLiteral(lits),
        });

        let mut loop_body = Vec::new();
        // get the current decision
        let current_decision = prefixvar(
            self.arena,
            "current_decision",
            VariableType::Extern("DecisionTreeNode"),
        );
        self.add_variable(current_decision.clone());
        loop_body.push(Statement::Assign {
            target: current_decision.clone(),
            value: Expression::ArrayAccess {
                array: Box::new(Expression::Variable(decision_tree_var.clone())),
                index: Box::new(Expression::Variable(decision_idx_var.clone())),
            },
        });

        // check if we should break the loop
        //let next_decision_idx_var = newvar(self.arena, "next_decision_idx", VariableType::I32);
        let location_var = prefixvar(self.arena, "location", VariableType::F32);
        //self.add_variable(next_decision_idx_var.clone());
        self.add_variable(location_var.clone());

        // decision_idx = next_decision_idx * (coord >= location) + (coord < location)
        loop_body.push(Statement::Assign {
            target: decision_idx_var.clone(),
            value: Expression::BinaryOp {
                op: BinaryOperator::Add,
                left: Box::new(Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(Expression::Field {
                        base: Box::new(Expression::Variable(current_decision.clone())),
                        field: "next_decision_index".into(),
                        type_of_field: VariableType::I32,
                        known_idnex: Some(1),
                    }),
                    right: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Less,
                        left: Box::new(Expression::Variable(first_coord_var.clone())),
                        right: Box::new(Expression::Field {
                            base: Box::new(Expression::Variable(current_decision.clone())),
                            field: "location".into(),
                            type_of_field: VariableType::F32,
                            known_idnex: Some(2),
                        }),
                    }),
                }),
                right: Box::new(Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Add,
                        left: Box::new(Expression::Variable(decision_idx_var.clone())),
                        right: Box::new(Expression::Int(1)),
                    }),
                    right: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::GreaterEqual,
                        left: Box::new(Expression::Variable(first_coord_var.clone())),
                        right: Box::new(Expression::Field {
                            base: Box::new(Expression::Variable(current_decision.clone())),
                            field: "location".into(),
                            type_of_field: VariableType::F32,
                            known_idnex: Some(2),
                        }),
                    }),
                }),
            },
        });

        loop_body.push(Statement::Assign {
            target: location_var.clone(),
            value: Expression::Field {
                base: Box::new(Expression::Variable(current_decision.clone())),
                field: "location".into(),
                type_of_field: VariableType::F32,
                known_idnex: Some(2),
            },
        });

        // if coord < location && decision_idx > max_decision_idx {
        // break
        // }

        // let cond = Expression::BinaryOp {
        //     op: BinaryOperator::LessEqual,
        //     left: Box::new(Expression::Variable(coord_var.clone())),
        //     right: Box::new(Expression::Variable(location_var.clone())),
        // };
        let cond = Expression::BinaryOp {
            op: BinaryOperator::And,
            left: Box::new(Expression::BinaryOp {
                op: BinaryOperator::Less,
                left: Box::new(Expression::Variable(first_coord_var.clone())),
                right: Box::new(Expression::Variable(location_var.clone())),
            }),
            right: Box::new(Expression::BinaryOp {
                op: BinaryOperator::GreaterEqual,
                left: Box::new(Expression::Variable(decision_idx_var.clone())),
                right: Box::new(Expression::Int(max_decision_idx as i32)),
            }),
        };

        let mut then_branch = Vec::new();
        then_branch.push(Statement::Break);
        loop_body.push(Statement::If {
            condition: cond,
            then_branch,
            else_branch: Vec::new(),
        });

        loop_body.push(Statement::Assign {
            target: coordinate_var.clone(),
            // value: Expression::Field {
            //     base: Box::new(Expression::Variable(current_decision.clone())),
            //     field: "next_coord".into(),
            //     type_of_field: VariableType::I32,
            // },
            value: Expression::ArrayAccess {
                array: Box::new(Expression::Variable(coordinates_array_var.clone())),
                index: Box::new(Expression::Field {
                    base: Box::new(Expression::Variable(current_decision.clone())),
                    field: "next_coord".into(),
                    type_of_field: VariableType::I32,
                    known_idnex: Some(0),
                }),
            },
        });

        self.add_statement(Statement::Repeat {
            count: max_iterations,
            body: loop_body,
        });

        // after the loop, we have the correct decision index to get the value and derivative
        self.add_statement(Statement::Assign {
            target: value_var.clone(),
            value: Expression::Field {
                base: Box::new(Expression::ArrayAccess {
                    array: Box::new(Expression::Variable(values_var.clone())),
                    index: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Subtract,
                        left: Box::new(Expression::Variable(decision_idx_var.clone())),
                        right: Box::new(Expression::Int(max_decision_idx as i32)),
                    }),
                }),
                field: "value".into(),
                type_of_field: VariableType::F32,
                known_idnex: Some(0),
            },
        });
        // self.add_statement(Statement::Assign {
        //     target: derivative_var.clone(),
        //     value: Expression::Field {
        //         base: Expression:: {
        //             array: Expression::Variable(values_var.clone()),
        //             index: Expression::BinaryOp {
        //                 op: BinaryOperator::Subtract,
        //                 left: Box::new(Expression::Variable(decision_idx_var.clone())),
        //                 right: Box::new(Expression::Int(max_iterations as i32)),
        //             },
        //         },
        //         field: "derivative".into(),
        //     },
        // });
        self.add_statement(Statement::Assign {
            target: derivative_var.clone(),
            value: Expression::Field {
                base: Box::new(Expression::ArrayAccess {
                    array: Box::new(Expression::Variable(values_var.clone())),
                    index: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Subtract,
                        left: Box::new(Expression::Variable(decision_idx_var.clone())),
                        right: Box::new(Expression::Int(max_decision_idx as i32)),
                    }),
                }),
                field: "derivative".into(),
                type_of_field: VariableType::F32,
                known_idnex: Some(1),
            },
        });

        // return the value (for now, just return the value without interpolation)
        Expression::Variable(value_var.clone())
    }

    fn add_decisions_recursive(&mut self, spline: &Spline<'a>) -> DecisionTree<'a> {
        let mut tree = DecisionTree {
            decisions: Vec::new(),
            values: Vec::new(),
            coordinates: Vec::new(),
            max_iterations: 0,
        };
        //let coord_input = self.lower_density_input(spline.coordinate, None);
        tree.coordinates.push(spline.coordinate);
        tree.decisions
            .push((spline.coordinate, 0, f64::NEG_INFINITY)); // extrapolation decision

        let sorted_points = {
            let mut points = Vec::from(spline.spline_points);
            points.sort_by(|a, b| a.location.partial_cmp(&b.location).unwrap());
            points
        };

        let mut to_add = Vec::new();
        let mut offset = spline.spline_points.len() as i32 + 1; // offset for decision indices of local trees
        for (i, point) in sorted_points.iter().enumerate() {
            // add decision for point.location
            // if point.value is a nested spline, build a local tree for it and add it to the decision tree
            match point.value {
                SplineValue::Const(c) => {
                    tree.values.push((c, point.derivative));
                    tree.decisions.push((
                        spline.coordinate,
                        -(tree.values.len() as i32 + 1), // negative index indicates a value index
                        point.location,
                    ));
                    let local_iterations = i + 1;
                    if local_iterations > tree.max_iterations {
                        tree.max_iterations = local_iterations;
                    }
                    offset += 1;
                }
                SplineValue::Spline(def) => {
                    let local_tree: DecisionTree<'_> = self.add_decisions_recursive(&def);
                    // add local tree decisions and values to the main tree
                    // for (var, next_decision_idx, location) in local_tree.decisions {
                    //     let n_index = if next_decision_idx >= 0 {
                    //         next_decision_idx + offset
                    //     } else {
                    //         next_decision_idx - v_offset as i32
                    //     }; // value indices are negative and should not be offset
                    //     tree.decisions.push((var, n_index, location));
                    // }
                    // for (value, derivative) in local_tree.values {
                    //     tree.values.push((value, derivative));
                    // }

                    // for coord in local_tree.coordinates {
                    //     tree.coordinates.push(coord);
                    // }

                    let local_iterations = local_tree.max_iterations + i + 1;
                    if local_iterations > tree.max_iterations {
                        tree.max_iterations = local_iterations;
                    }
                    let tree_decisions_len = tree.decisions.len() as i32;
                    to_add.push(local_tree);

                    tree.decisions.push((
                        spline.coordinate,
                        offset + 1, // point to the first decision of the local tree
                        point.location,
                    ));
                    offset += tree_decisions_len;
                }
            }
        }

        // add final extrapolation decision for input >= last point
        let (_, ndi, _) = tree.decisions.last().unwrap();
        tree.decisions
            .push((spline.coordinate, *ndi, f64::INFINITY));
        tree.max_iterations += 1;

        // add local trees after the main loop to maintain correct offsets
        for local_tree in to_add {
            let offset = tree.coordinates.len() as i32;
            let v_offset = tree.values.len();
            for (var, next_decision_idx, location) in local_tree.decisions {
                let n_index = if next_decision_idx >= 0 {
                    next_decision_idx + offset
                } else {
                    next_decision_idx - v_offset as i32
                }; // value indices are negative and should not be offset
                tree.decisions.push((var, n_index, location));
            }
            for (value, derivative) in local_tree.values {
                tree.values.push((value, derivative));
            }

            for coord in local_tree.coordinates {
                tree.coordinates.push(coord);
            }
        }

        tree
    }

    fn make_decision_tree_init(&mut self, decisions: &[(u8, u32, f64)]) -> Expression<'m> {
        // convert the decisions to an array literal
        let elements = decisions
            .iter()
            .map(
                |(coord_idx, next_decision_idx, location)| Expression::ConstructExtern {
                    t: VariableType::Extern("DecisionTreeNode"),
                    args: vec![
                        ("next_coord", Expression::Int(*coord_idx as i32)),
                        (
                            "next_decision_index",
                            Expression::Int(*next_decision_idx as i32),
                        ),
                        ("location", Expression::Float(*location as f32)),
                    ],
                },
            )
            .collect();
        Expression::ArrayLiteral(elements)
    }

    fn make_values_init(&mut self, values: &[(f64, f64)]) -> Expression<'m> {
        // convert the values to an array literal
        let elements = values
            .iter()
            .map(|(value, derivative)| Expression::ConstructExtern {
                t: VariableType::Extern("SplineValue"),
                args: vec![
                    ("value", Expression::Float(*value as f32)),
                    ("derivative", Expression::Float(*derivative as f32)),
                ],
            })
            .collect();
        Expression::ArrayLiteral(elements)
    }
}
