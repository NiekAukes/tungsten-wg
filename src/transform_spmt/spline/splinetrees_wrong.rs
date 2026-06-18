use crate::{
    parse::model::{Density, Spline, SplinePoint, SplineValue},
    spmt::model::{BinaryOperator, Expression, Function, Statement, Var, Variable, VariableType},
    transform_spmt::{density::DensityBuilder, newvar, prefixvar},
};

#[derive(Debug, Clone)]
struct InterpolationLevel {
    index: usize,     // the tree index of this interpolation
    derivative1: f64, // the derivative at the first point
    derivative2: f64, // the derivative at the second point
    location1: f64,   // the location of the first point
    location2: f64,   // the location of the second point
}

#[derive(Debug, Clone)]
struct InterpolationChain {
    points: Vec<InterpolationLevel>, // levels of interpolation, each level has two points and their derivatives
}

impl InterpolationLevel {
    fn get_level(&self) -> usize {
        // k=⌊log2​(i+1)⌋
        ((point.index + 1) as f64).log2().floor() as usize
    }
}

impl InterpolationChain {
    fn new() -> Self {
        InterpolationChain { points: Vec::new() }
    }

    fn add_level(&mut self, level: InterpolationLevel) {
        self.points.push(level);
    }

    fn move_down_to_left(&mut self) {
        // change the indices such that index 0 becomes 1
        // index 1 becomes 3
        // index 2 becomes 4
        // index 3 becomes 7, etc. (index n becomes 2^(n+1) - 1)
        for point in self.points.iter_mut() {
            let k = point.get_level();
            point.index = (point.index * 2.pow(k))
        }
    }

    fn move_down_to_right(&mut self) {
        for point in self.points.iter_mut() {
            let k = point.get_level();
            point.index = (point.index * 2.pow(k+1))
        }
    }
    
}

struct DecisionTree<'a> {
    decisions: Vec<(Density<'a>, i32, f64)>, // next_coord_idx, next_decision_idx, next_value_idx location
    //values: Vec<(f64, f64)>,                 // (value, derivative)
    interp_chains: Vec<InterpolationChain>,
    coordinates: Vec<Density<'a>>,
    max_iterations: usize,
}

impl<'a> DecisionTree<'a> {
    fn new() -> Self {
        DecisionTree {
            decisions: Vec::new(),
            interp_chains: Vec::new(),
            coordinates: Vec::new(),
            max_iterations: 0,
        }
    }

    fn finish(
        self,
    ) -> (
        Vec<(u8, u32, f64)>,
        Vec<InterpolationChain>,
        Vec<Density<'a>>,
    ) {
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
        (decisions, self.interp_chains, inputs)
    }
}

impl<'a, 'm> DensityBuilder<'a, 'm> {
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
        let mut tree = DecisionTree::new();
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
        let mut decisions_taken = Vec::new();
        let mut moving_chain_index = 0;
        let mut values = Vec::new();
        for (i, point) in sorted_points.iter().enumerate() {
            // add decision for point.location
            // if point.value is a nested spline, build a local tree for it and add it to the decision tree
            match point.value {
                SplineValue::Const(c) => {
                    //tree.values.push((c, point.derivative));
                    decisions_taken.push(i);
                    let chain = self.add_interpolation_chains_for_endpoints(tree, &decisions_taken);
                    tree.interp_chains.push(chain);
                    tree.decisions.push((
                        spline.coordinate,
                        -(tree.interp_chains.len() as i32 + 1), // negative index indicates a value index
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

    fn add_interpolation_chains_for_endpoints(&mut self, spline: &Spline<'a>, route: &[usize]) -> InterpolationChain {
        // walk the entire decision route and add interpolation levels for each endpoint
        let (first, remainder) = route.split_first().unwrap();
        let point = &spline.spline_points[*first];
        if *first == 0 {
            // no recursion, just add the point as an interpolation level
            // this is actually an extrapolation level, but we can treat it as an interpolation between infinity and the first point
            let mut lower_chain = self.add_interpolation_chains_for_endpoints(&spline, remainder);
            lower_chain.move_down_to_right();

            // this interpolation represents a straight line, i.e. extrapolation
            lower_chain.add_level(InterpolationLevel {
                index: 0,
                derivative1: point.0.derivative, // derivative at the first point
                derivative2: point.0.derivative, // derivative at the first point
                location1: -1_000_000.0, // use a very large negative number to represent negative infinity
                location2: point.2,
            });
            return lower_chain;
        }

        // if the point is +infinity, extrapolate the other way
        if point.location.is_infinite() && point.location.is_sign_positive() {
            let point_minus_one = &spline.spline_points[];
            let mut upper_chain = self.add_interpolation_chains_for_endpoints(&spline, remainder);
            upper_chain.move_down_to_left();

            // this interpolation represents a straight line, i.e. extrapolation
            upper_chain.add_level(InterpolationLevel {
                index: 0,
                derivative1: point.0.derivative, // derivative at the last point
                derivative2: point.0.derivative, // derivative at the last point
                location1: 1_000_000.0, // use a very large positive number to represent positive infinity
                location2: point.2,
            });
            return upper_chain;
        }
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
