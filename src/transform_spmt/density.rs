use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
    result,
};

use crate::{
    orchestrate::Scale, parse::model::{Density, DensityType, NormalNoise}, spmt::model::{
        BinaryOperator, DensityFunction, DensityFunctionRef, DensityInput, Expression, Function,
        FunctionRef, SPMT, Statement, Variable, VariableType,
    }, transform_spmt::{
        BuilderState, DensityFunctionCache, NoiseCache, anonvar, newvar, noise::lower_normal_noise,
    }
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormalNoiseScaled<'a> {
    noise: NormalNoise<'a>,
    scale: Scale,
}

/// Builds a single density function for a given density.
/// This is used to build the density functions for the model,
/// Statements can be added to the builder
pub struct DensityBuilder<'a, 'm> {
    density_function: DensityFunction<'m>,
    function: Option<Function<'m>>,
    pub arena: &'m bumpalo::Bump,

    builder_state: Option<BuilderState<'a, 'm>>,

    density_function_inputs: HashMap<Density<'a>, DensityInput<'m>>,
    noise_inputs: std::collections::HashMap<NormalNoiseScaled<'a>, DensityInput<'m>>,
    helper_functions: Vec<FunctionRef<'m>>,
    pub(crate) origin: Rc<Variable>,
    pub(crate) rpos3: Rc<Variable>,
    pub(crate) p: Rc<Variable>,
}

impl<'a, 'm> DensityBuilder<'a, 'm> {
    pub fn new(arena: &'m bumpalo::Bump, state: BuilderState<'a, 'm>) -> Self {
        let mut s = Self {
            density_function: DensityFunction {
                body: Vec::new(),
                canonical_name: None,
                density_inputs: Vec::new(),
                variables: Vec::new(),
                helper_functions: Vec::new(),
            },
            function: None,
            helper_functions: Vec::new(),
            arena,

            builder_state: Some(state),
            density_function_inputs: HashMap::new(),
            noise_inputs: HashMap::new(),
            p: newvar("pos3", VariableType::Pos3),
            rpos3: newvar("rpos3", VariableType::Vec3),
            origin: newvar("origin", VariableType::Vec3),
        };

        s.add_variable(s.rpos3.clone());

        // initialize rpos3 = origin + p * scaled_origin
        s.add_statement(Statement::Assign {
            target: s.rpos3.clone(),
            value: make_rpos3(
                s.p.clone(),
                s.origin.clone(),
                s.builder_state.as_ref().unwrap().working_scaled_origin,
            ),
        });
        s
    }

    pub fn new_named(
        arena: &'m bumpalo::Bump,
        state: BuilderState<'a, 'm>,
        canonical_name: Option<String>,
    ) -> Self {
        let mut s = Self::new(arena, state);
        s.density_function.canonical_name = canonical_name;
        s
    }

    pub fn switch_function(&mut self, function: Function<'m>) -> Option<Function<'m>> {
        let old = self.function.replace(function);
        old
    }

    pub fn finish(
        self,
        ret: Expression<'m>,
    ) -> (
        DensityFunction<'m>,
        Vec<FunctionRef<'m>>,
        BuilderState<'a, 'm>,
    ) {
        let mut function = self.density_function;
        function.helper_functions = self.helper_functions.clone();
        function.add_statement(Statement::Return(ret));

        // if the function name is None, generate a unique name based on the address of the function
        if function.canonical_name.is_none() {
            let a = self.arena.alloc(());
            let id = a as *const () as usize;
            function.canonical_name = Some(format!("density_function_{}", id));
        }

        (
            function,
            self.helper_functions,
            self.builder_state
                .expect("Builder state is not initialized"),
        )
    }

    pub fn finish_and_continue(
        &mut self,
        ret: Expression<'m>,
        replacement: Option<Function<'m>>,
    ) -> FunctionRef<'m> {
        let mut function = self.function.take().expect("No function to finish");
        function.add_statement(Statement::Return(ret));

        let function_ref = FunctionRef::new(self.arena.alloc(function));
        self.helper_functions.push(function_ref);

        if let Some(replacement) = replacement {
            self.function = Some(replacement);
        }

        function_ref
    }

    pub fn add_statement(&mut self, statement: Statement<'m>) {
        match &mut self.function {
            Some(func) => func.body.push(statement),
            None => self.density_function.body.push(statement),
        }
    }

    pub fn add_variable(&mut self, variable: Rc<Variable>) {
        match &mut self.function {
            Some(func) => func.variables.push(variable),
            None => self.density_function.variables.push(variable),
        }
    }

    pub fn lower_noise(&mut self, density: NormalNoise<'a>, scale: (f32, f32, f32)) -> FunctionRef<'m> {
        // lower the noise into a density function
        let function = lower_normal_noise(density, None, scale, false);
        let function = FunctionRef::new(self.arena.alloc(function));
        function
    }

    pub fn lower_noise_as_density(&mut self, density: NormalNoise<'a>, scale: (f32, f32, f32)) -> DensityFunctionRef<'m> {
        // lower the noise into a density function
        let function = lower_normal_noise(density, None,scale, true);
        let mut density_function = DensityFunction {
            body: function.body,
            canonical_name: function.canonical_name,
            density_inputs: vec![],
            variables: function.variables,
            helper_functions: vec![],
        };

        if density_function.canonical_name.is_none() {
            // allocate some bytes in the arena to get a unique id for the function
            let a = self.arena.alloc(());
            let id = a as *const () as usize;
            density_function.canonical_name = Some(format!("noise_function_{}", id));
        }

        let density_function_ref = DensityFunctionRef::new(self.arena.alloc(density_function));
        density_function_ref
    }

    pub fn lower_noise_and_mark(&mut self, noise: NormalNoise<'a>, xz_scale: f64, y_scale: f64) -> DensityInput<'m> {
        // check if we already have a density function for this noise
        // create a scaled noise struct to use as a key for the cache
        let noise_scaled = NormalNoiseScaled {
            noise,
            scale: Scale::new(xz_scale as f32, y_scale as f32, xz_scale as f32),
        };
        if let Some(cached) = self.noise_inputs.get(&noise_scaled) {
            return cached.clone();
        }

        // borrow the caches from the builder state
        let mut bs = self.builder_state.take().unwrap();
        // lower the noise into a density function
        let density_function_ref = if let Some(cached) = bs.noise_cache.get(&noise) {
            cached.clone()
        } else {
            let density_function_ref = self.lower_noise_as_density(noise,
                (xz_scale as f32, y_scale as f32, xz_scale as f32));
            bs.noise_cache.insert(noise, density_function_ref.clone());
            density_function_ref
        };
        let v = anonvar(VariableType::DensityInput);
        let dimensions = bs.working_dimensions;
        let mut scaled_origin = bs.working_scaled_origin;
        scaled_origin.0 *= xz_scale as f32;
        scaled_origin.1 *= y_scale as f32;
        scaled_origin.2 *= xz_scale as f32;
        let input = DensityInput {
            var: v.clone(),
            density_function: density_function_ref.clone(),
            scaled_origin: scaled_origin,
            dimensions: dimensions,
        };

        self.noise_inputs.insert(noise_scaled, input.clone());
        self.density_function.density_inputs.push(input.clone());
        if let Some(func) = &mut self.function {
            func.variables.push(v);
        }

        // return the density input, and put back the caches into the builder state
        self.builder_state = Some(bs);
        input
    }

    pub fn _lower_density_shader_inner(
        &mut self,
        density: Density<'a>,
        canonical_name: Option<String>,
    ) -> DensityFunctionRef<'m> {
        // borrow the caches from the builder state
        let mut bs = self.builder_state.take().unwrap();
        // lower the noise into a density function
        let density_function_ref = if let Some(cached) = bs.density_function_cache.get(&density) {
            cached.clone()
        } else {
            // lower the density into an expression
            // create a new builder to build the density function
            let mut builder = DensityBuilder::new_named(self.arena, bs, canonical_name);
            let r = builder.lower_density(density);
            let (density_function, helpers, bs_returned) = builder.finish(r);
            //self.helper_functions.extend(helpers);
            bs = bs_returned;

            let density_function_ref = DensityFunctionRef::new(self.arena.alloc(density_function));
            bs.density_function_cache
                .insert(density, density_function_ref.clone());
            density_function_ref
        };

        // return the density input, and put back the caches into the builder state
        self.builder_state = Some(bs);
        density_function_ref
    }

    pub fn lower_density_input(
        &mut self,
        density: Density<'a>,
        canonical_name: Option<String>,
    ) -> DensityInput<'m> {
        // check if we already have a density function for this density
        if let Some(cached) = self.density_function_inputs.get(&density) {
            return cached.clone();
        }

        let density_function_ref = self._lower_density_shader_inner(density, canonical_name);

        let dimensions = self.builder_state.as_ref().unwrap().working_dimensions;
        let scaled_origin = self.builder_state.as_ref().unwrap().working_scaled_origin;
        let v = anonvar(VariableType::DensityInput);
        let input = DensityInput {
            var: v.clone(),
            density_function: density_function_ref.clone(),
            scaled_origin: scaled_origin,
            dimensions: dimensions,
        };

        self.density_function_inputs.insert(density, input.clone());
        self.density_function.density_inputs.push(input.clone());
        if let Some(func) = &mut self.function {
            func.variables.push(v);
        }

        input
    }

    pub fn lower_density_input_expanded(
        &mut self,
        density: Density<'a>,
        canonical_name: Option<String>,
        dimensions: (i32, i32, i32),
        scaled_origin: (f32, f32, f32),
    ) -> DensityInput<'m> {
        // check if we already have a density function for this density
        if let Some(cached) = self.density_function_inputs.get(&density) {
            return cached.clone();
        }

        let mut bs = self.builder_state.take().unwrap();
        let old_dimensions = bs.working_dimensions;
        let old_scaled_origin = bs.working_scaled_origin;

        bs.working_dimensions = dimensions;
        bs.working_scaled_origin = scaled_origin;
        self.builder_state = Some(bs);

        let density_function_ref = self._lower_density_shader_inner(density, canonical_name);

        let v = anonvar(VariableType::DensityInput);
        let input = DensityInput {
            var: v.clone(),
            density_function: density_function_ref.clone(),
            scaled_origin: scaled_origin,
            dimensions: dimensions,
        };

        self.density_function_inputs.insert(density, input.clone());
        self.density_function.density_inputs.push(input.clone());
        if let Some(func) = &mut self.function {
            func.variables.push(v);
        }

        // restore the old dimensions and scaled origin
        let mut bs = self.builder_state.take().unwrap();
        bs.working_dimensions = old_dimensions;
        bs.working_scaled_origin = old_scaled_origin;
        self.builder_state = Some(bs);
        input
    }

    pub fn lower_density(&mut self, density: Density<'a>) -> Expression<'m> {
        match *density {
            DensityType::Noise { noise, xz_scale, y_scale } => {
                // A noise call specifically
                let fref = self.lower_noise_and_mark(noise, xz_scale, y_scale);
                // use the density input as the expression
                Expression::DensityVariable(fref)
            }
            DensityType::Const(v) => Expression::Float(v as f64),
            DensityType::Add { left, right } => {
                let left = self.lower_density(left);
                let right = self.lower_density(right);
                Expression::BinaryOp {
                    op: BinaryOperator::Add,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            DensityType::Multiply { left, right } => {
                let left = self.lower_density(left);
                let right = self.lower_density(right);
                Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            DensityType::Cache2d { argument } => {
                // do nothing for now, just lower the argument
                // eventually we might want to mark this as a special function call, so that we can do some caching later on
                self.lower_density(argument)
            }
            DensityType::Squeeze { argument } => {
                /*
                x = clamp(argument, -1, 1)
                result = x/2 - x*x*x/24
                return result
                */
                let xv = anonvar(VariableType::F32);
                let arg_expr = self.lower_density(argument);
                self.add_variable(xv.clone());
                self.add_statement(Statement::Assign {
                    target: xv.clone(),
                    value: make_clamp(arg_expr, -1.0, 1.0),
                });
                let result = Expression::BinaryOp {
                    op: BinaryOperator::Subtract,
                    left: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Multiply,
                        left: Box::new(Expression::Variable(xv.clone())),
                        right: Box::new(Expression::Float(0.5)),
                    }),
                    right: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Multiply,
                        left: Box::new(Expression::BinaryOp {
                            op: BinaryOperator::Multiply,
                            left: Box::new(Expression::Variable(xv.clone())),
                            right: Box::new(Expression::Variable(xv.clone())),
                        }),
                        right: Box::new(Expression::Float(1.0 / 24.0)),
                    }),
                };
                let result_var = anonvar(VariableType::F32);
                self.add_variable(result_var.clone());
                self.add_statement(Statement::Assign {
                    target: result_var.clone(),
                    value: result,
                });
                Expression::Variable(result_var)
            }
            DensityType::Interpolated { argument } => {
                // TODO: another optimization
                // skip for now
                self.lower_density(argument)
            }
            DensityType::EndIslands => {
                // this is a special density function that doesn't take any arguments, it just marks the end of island generation
                // for now we can just return 0, since this density function is only used in a condition to check if we should generate an island or not
                Expression::Float(0.0)
            }
            DensityType::YClampedGradient {
                from_y,
                to_y,
                from_value,
                to_value,
            } => {
                // 1. Access p.y
                let p_var = Expression::Variable(self.p.clone());

                let y_expr = Expression::Field {
                    base: Box::new(p_var),
                    field: "y".to_string(),
                };

                // 2. clamp(p.y, from_y, to_y)
                let clamped_expr = Expression::ExternCall {
                    function_name: "clamp".to_string(),
                    parameters: vec![y_expr, Expression::Float(from_y), Expression::Float(to_y)],
                    parameter_types: vec![VariableType::F32, VariableType::F32, VariableType::F32],
                };

                // 3. Create clampedY variable
                let clamped_y = anonvar(VariableType::F32);
                self.add_variable(clamped_y.clone());

                self.add_statement(Statement::Assign {
                    target: clamped_y.clone(),
                    value: clamped_expr,
                });

                // ---- Build linear mapping ----

                // (clampedY - from_y)
                let numerator = Expression::BinaryOp {
                    op: BinaryOperator::Subtract,
                    left: Box::new(Expression::Variable(clamped_y.clone())),
                    right: Box::new(Expression::Float(from_y)),
                };

                // (to_y - from_y)
                let denominator = Expression::BinaryOp {
                    op: BinaryOperator::Subtract,
                    left: Box::new(Expression::Float(to_y)),
                    right: Box::new(Expression::Float(from_y)),
                };

                // (clampedY - from_y) / (to_y - from_y)
                let normalized = Expression::BinaryOp {
                    op: BinaryOperator::Divide,
                    left: Box::new(numerator),
                    right: Box::new(denominator),
                };

                // (to_value - from_value)
                let value_range = Expression::BinaryOp {
                    op: BinaryOperator::Subtract,
                    left: Box::new(Expression::Float(to_value)),
                    right: Box::new(Expression::Float(from_value)),
                };

                // normalized * value_range
                let scaled = Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(normalized),
                    right: Box::new(value_range),
                };

                // + from_value
                Expression::BinaryOp {
                    op: BinaryOperator::Add,
                    left: Box::new(scaled),
                    right: Box::new(Expression::Float(from_value)),
                }
            }
            DensityType::FlatCache { argument } => {
                // another caching function, for now just ignore the caching and lower the argument
                // TODO: implement caching
                if let Some(cached) = self.density_function_inputs.get(&density) {
                    return Expression::DensityVariable(cached.clone());
                }

                // flat caches are always lowered as separate density functions
                let input = self.lower_density_input(argument, None);
                // add to cache
                self.density_function_inputs.insert(density, input.clone());
                // return the density variable for the input
                Expression::DensityVariable(input)
            }
            DensityType::NamedDensityReference { argument, name } => {
                // another caching function, for now just ignore the caching and lower the argument
                // TODO: implement caching
                if let Some(cached) = self.density_function_inputs.get(&density) {
                    return Expression::DensityVariable(cached.clone());
                }
                let name = name.clone();
                // flat caches are always lowered as separate density functions
                let input = self.lower_density_input(argument, Some(name));
                // add to cache
                self.density_function_inputs.insert(density, input.clone());
                // return the density variable for the input
                Expression::DensityVariable(input)
            }
            DensityType::OldBlendedNoise {
                smear_scale_multiplier,
                xz_factor,
                xz_scale,
                y_factor,
                y_scale,
            } => {
                // old noise function, used in 1.18 and earlier
                // for now call an extern function by the same name
                Expression::ExternCall {
                    function_name: "old_blended_noise".into(),
                    parameters: vec![
                        Expression::Variable(self.rpos3.clone()),
                        Expression::Float(smear_scale_multiplier),
                        Expression::Float(xz_factor),
                        Expression::Float(xz_scale),
                        Expression::Float(y_factor),
                        Expression::Float(y_scale),
                    ],
                    parameter_types: vec![
                        VariableType::Vec3,
                        VariableType::F32,
                        VariableType::F32,
                        VariableType::F32,
                        VariableType::F32,
                        VariableType::F32,
                    ],
                }
            }
            DensityType::ShiftedNoise {
                noise,
                shift_x,
                shift_y,
                shift_z,
                xz_scale,
                y_scale,
            } => {
                // 1. Lower shift densities
                let shift_x_expr = self.lower_density(shift_x);
                let shift_y_expr = Expression::Float(shift_y); // shift_y is f64 constant
                let shift_z_expr = self.lower_density(shift_z);

                // 2. Build vec3(shiftX, shiftY, shiftZ)
                let shift_vec = Expression::Construct {
                    t: VariableType::Vec3,
                    args: vec![shift_x_expr, shift_y_expr, shift_z_expr],
                };

                // 3. p + shift_vec
                let shifted_position = Expression::BinaryOp {
                    op: BinaryOperator::Add,
                    left: Box::new(Expression::Variable(self.rpos3.clone())),
                    right: Box::new(shift_vec),
                };

                // 4. Lower noise but don't mark it, we just want the density function reference
                let noise_function_ref = self.lower_noise(noise, 
                    (xz_scale as f32, y_scale as f32, xz_scale as f32));
                // add it as a helper function
                self.helper_functions.push(noise_function_ref.clone());

                Expression::FunctionCall {
                    function: noise_function_ref,
                    parameters: vec![shifted_position.into()],
                }
            }
            DensityType::ShiftA { argument } => {
                // Samples a noise at (x/4, 0, z/4), then multiplies it by 4.
                let x_shift = Expression::BinaryOp {
                    op: BinaryOperator::Divide,
                    left: Box::new(Expression::Field {
                        base: Box::new(Expression::Variable(self.p.clone())),
                        field: "x".to_string(),
                    }),
                    right: Box::new(Expression::Float(4.0)),
                };
                let z_shift = Expression::BinaryOp {
                    op: BinaryOperator::Divide,
                    left: Box::new(Expression::Field {
                        base: Box::new(Expression::Variable(self.p.clone())),
                        field: "z".to_string(),
                    }),
                    right: Box::new(Expression::Float(4.0)),
                };
                let shift_vec = Expression::Construct {
                    t: VariableType::Pos3,
                    args: vec![x_shift, Expression::Float(0.0), z_shift],
                };

                // create a new builder to build the argument density, we just want to lower the argument density to get the density function reference, we don't care about the body or variables of the builder
                // let bs = self.builder_state.take().unwrap();
                // let mut argument_builder = DensityBuilder::new(self.arena, bs);
                // let r = argument_builder.lower_density(argument);
                // let (argument_density_function, helpers, bs) = argument_builder.finish(r);
                // self.builder_state = Some(bs);
                // self.helper_functions.extend(helpers);
                // let argument_density_function_ref =
                //     DensityFunctionRef::new(self.arena.alloc(argument_density_function));
                let dimensions = self.builder_state.as_ref().unwrap().working_dimensions;
                let dimensions = (dimensions.0, 1, dimensions.2);
                let scaled_origin = self.builder_state.as_ref().unwrap().working_scaled_origin;
                let scaled_origin = (scaled_origin.0 / 4.0, 0.0, scaled_origin.2 / 4.0);
                let density_input =
                    self.lower_density_input_expanded(argument, None, dimensions, scaled_origin);

                // let call = Expression::DensityFunctionCall {
                //     function: argument_density_function_ref,
                //     position: shift_vec.into(),
                // };
                let call = Expression::DensityVariable(density_input);
                // multiply the result by 4
                Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(call),
                    right: Box::new(Expression::Float(4.0)),
                }
            }
            DensityType::ShiftB { argument } => {
                // Samples a noise at (x/4, y/4, 0), then multiplies it by 4.
                // let shift_vec = Expression::MakeVec3 {
                //     x: Box::new(Expression::BinaryOp {
                //         op: BinaryOperator::Divide,
                //         left: Box::new(Expression::Field {
                //             base: Box::new(Expression::Variable(self.p.clone())),
                //             field: "x".to_string(),
                //         }),
                //         right: Box::new(Expression::Float(4.0)),
                //     }),
                //     y: Box::new(Expression::BinaryOp {
                //         op: BinaryOperator::Divide,
                //         left: Box::new(Expression::Field {
                //             base: Box::new(Expression::Variable(self.p.clone())),
                //             field: "y".to_string(),
                //         }),
                //         right: Box::new(Expression::Float(4.0)),
                //     }),
                //     z: Box::new(Expression::Float(0.0)),
                // };
                let x_shift = Expression::BinaryOp {
                    op: BinaryOperator::Divide,
                    left: Box::new(Expression::Field {
                        base: Box::new(Expression::Variable(self.p.clone())),
                        field: "x".to_string(),
                    }),
                    right: Box::new(Expression::Float(4.0)),
                };

                let y_shift = Expression::BinaryOp {
                    op: BinaryOperator::Divide,
                    left: Box::new(Expression::Field {
                        base: Box::new(Expression::Variable(self.p.clone())),
                        field: "y".to_string(),
                    }),
                    right: Box::new(Expression::Float(4.0)),
                };

                let shift_vec = Expression::Construct {
                    t: VariableType::Pos3,
                    args: vec![x_shift, y_shift, Expression::Float(0.0)],
                };

                let density_input = self.lower_density_input(argument, None);
                let call = Expression::DensityVariable(density_input);

                // multiply the result by 4
                Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(call),
                    right: Box::new(Expression::Float(4.0)),
                }
            }
            DensityType::CacheOnce { argument } => {
                // another caching function, for now just ignore the caching and lower the argument
                self.lower_density(argument)
            }
            DensityType::Spline { spline } => {
                let f = self.lower_spline_definition(spline, None);
                let fref = self.arena.alloc(f);
                let fref = FunctionRef::new(fref);
                Expression::FunctionCall {
                    function: fref,
                    parameters: vec![Expression::Variable(self.p.clone())],
                }
            }
            DensityType::Abs { argument } => {
                let arg_expr = self.lower_density(argument);
                Expression::ExternCall {
                    function_name: "abs".into(),
                    parameters: vec![arg_expr],
                    parameter_types: vec![VariableType::F32],
                }
            }
            DensityType::Min { left, right } => {
                let left_expr = self.lower_density(left);
                let right_expr = self.lower_density(right);
                Expression::ExternCall {
                    function_name: "min".into(),
                    parameters: vec![left_expr, right_expr],
                    parameter_types: vec![VariableType::F32, VariableType::F32],
                }
            }
            DensityType::Max { left, right } => {
                let left_expr = self.lower_density(left);
                let right_expr = self.lower_density(right);
                Expression::ExternCall {
                    function_name: "max".into(),
                    parameters: vec![left_expr, right_expr],
                    parameter_types: vec![VariableType::F32, VariableType::F32],
                }
            }
            DensityType::RangeChoice {
                input,
                min_inclusive,
                max_exclusive,
                when_in_range,
                when_out_of_range,
            } => {
                let input = self.lower_density(input);
                let in_range_expr = self.lower_density(when_in_range);
                let out_of_range_expr = self.lower_density(when_out_of_range);

                let v = anonvar(VariableType::F32);
                self.add_variable(v.clone());
                self.add_statement(Statement::Assign {
                    target: v.clone(),
                    value: input.clone(),
                });

                let condition = Expression::BinaryOp {
                    op: BinaryOperator::And,
                    left: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::GreaterEqual,
                        left: Box::new(Expression::Variable(v.clone())),
                        right: Box::new(Expression::Float(min_inclusive)),
                    }),
                    right: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Less,
                        left: Box::new(Expression::Variable(v.clone())),
                        right: Box::new(Expression::Float(max_exclusive)),
                    }),
                };

                let v = anonvar(VariableType::F32);
                self.add_variable(v.clone());
                self.add_statement(Statement::If {
                    condition,
                    then_branch: vec![Statement::Assign {
                        target: v.clone(),
                        value: in_range_expr,
                    }],
                    else_branch: vec![Statement::Assign {
                        target: v.clone(),
                        value: out_of_range_expr,
                    }],
                });

                Expression::Variable(v)
            }
            DensityType::Clamp { input, min, max } => {
                let input = self.lower_density(input);
                make_clamp(input, min, max)
            }
            DensityType::WeirdScaledSampler {
                input,
                noise_to_sample,
                ref rarity_value_mapper,
            } => {
                // TODO: implement this
                // for now just lower the input density and ignore the weird sampling
                self.lower_density(input)
            }
            DensityType::Square { argument } => {
                let v = anonvar(VariableType::F32);
                let arg_expr = self.lower_density(argument);
                self.add_variable(v.clone());
                self.add_statement(Statement::Assign {
                    target: v.clone(),
                    value: arg_expr,
                });
                Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(Expression::Variable(v.clone())),
                    right: Box::new(Expression::Variable(v.clone())),
                }
            }
            DensityType::Cube { argument } => {
                let v = anonvar(VariableType::F32);
                let arg_expr = self.lower_density(argument);
                self.add_variable(v.clone());
                self.add_statement(Statement::Assign {
                    target: v.clone(),
                    value: arg_expr,
                });
                Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(Expression::Variable(v.clone())),
                    right: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Multiply,
                        left: Box::new(Expression::Variable(v.clone())),
                        right: Box::new(Expression::Variable(v.clone())),
                    }),
                }
            }
        }
    }
}

pub fn make_clamp(input: Expression, min: f64, max: f64) -> Expression {
    // clamp(x, min, max) = max(min, min(x, max))
    // Expression::BinaryOp {
    //     op: BinaryOperator::Max,
    //     left: Box::new(Expression::Literal(min)),
    //     right: Box::new(Expression::BinaryOp {
    //         op: BinaryOperator::Min,
    //         left: Box::new(input),
    //         right: Box::new(Expression::Literal(max)),
    //     }),
    // }
    Expression::ExternCall {
        function_name: "clamp".into(),
        parameters: vec![input, Expression::Float(min), Expression::Float(max)],
        parameter_types: vec![VariableType::F32, VariableType::F32, VariableType::F32],
    }
}

pub fn make_rpos3<'m>(
    p: Rc<Variable>,
    origin: Rc<Variable>,
    scale: (f32, f32, f32),
) -> Expression<'m> {
    // rpos3 = origin + p * scale
    if scale == (1.0, 1.0, 1.0) {
        return Expression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(Expression::Variable(origin)),
            right: Box::new(Expression::Variable(p)),
        };
    }
    Expression::BinaryOp {
        op: BinaryOperator::Add,
        left: Box::new(Expression::Variable(origin)),
        right: Box::new(Expression::BinaryOp {
            op: BinaryOperator::Multiply,
            left: Box::new(Expression::Variable(p)),
            right: Box::new(Expression::Construct {
                t: VariableType::Vec3,
                args: vec![
                    Expression::Float(scale.0 as f64),
                    Expression::Float(scale.1 as f64),
                    Expression::Float(scale.2 as f64),
                ],
            }),
        }),
    }
}
