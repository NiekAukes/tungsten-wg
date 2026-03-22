use std::collections::{HashMap, HashSet};

use crate::{
    orchestrate::Scale,
    parse::model::{Density, DensityType, NormalNoise},
    spmt::model::{
        BinaryOperator, DensityFunction, DensityFunctionRef, DensityInput, Expression, Function,
        FunctionRef, PermutationTableInput, SPMT, Statement, Var, Variable, VariableType,
    },
    transform_spmt::{
        BuilderState, DensityFunctionCache, NoiseCache, anonvar, newvar, noise::lower_normal_noise,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormalNoiseScaled<'a> {
    noise: NormalNoise<'a>,
    name: String,
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
    pub(crate) origin: Var<'m>,
    pub(crate) rpos3: Var<'m>,
    pub(crate) p: Var<'m>,
}

impl<'a, 'm> DensityBuilder<'a, 'm> {
    pub fn new(arena: &'m bumpalo::Bump, state: BuilderState<'a, 'm>) -> Self {
        let mut s = Self {
            density_function: DensityFunction {
                body: Vec::new(),
                canonical_name: None,
                density_inputs: Vec::new(),
                permutation_table_inputs: Vec::new(),
                variables: Vec::new(),
                helper_functions: Vec::new(),
            },
            function: None,
            helper_functions: Vec::new(),
            arena,

            builder_state: Some(state),
            density_function_inputs: HashMap::new(),
            noise_inputs: HashMap::new(),
            p: newvar(arena, "pos3", VariableType::Pos3),
            rpos3: newvar(arena, "rpos3", VariableType::Vec3),
            origin: newvar(arena, "origin", VariableType::Vec3),
        };

        s.add_variable(s.rpos3.clone());

        // initialize rpos3 = origin + p * scaled_origin
        s.add_statement(Statement::Assign {
            target: s.rpos3.clone(),
            value: make_rpos3(
                s.p.clone(),
                s.origin.clone(),
                s.builder_state.as_ref().unwrap().working_scaled_origin,
                s.builder_state.as_ref().unwrap().working_scaled_position,
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

    pub fn add_variable(&mut self, variable: Var<'m>) {
        match &mut self.function {
            Some(func) => func.variables.push(variable),
            None => self.density_function.variables.push(variable),
        }
    }

    pub fn lower_noise(
        &mut self,
        density: NormalNoise<'a>,
        origin_scale: (f32, f32, f32),
        position_scale: (f32, f32, f32),
        permutation_name: &str,
        name: String,
    ) -> (FunctionRef<'m>, Vec<PermutationTableInput>) {
        // lower the noise into a density function
        let (mut function, perm_tables) = lower_normal_noise(
            self.arena,
            density,
            permutation_name,
            name,
            origin_scale,
            position_scale,
            false,
        );
        // add the permutation tables to the function arguments
        for perm_table in &perm_tables {
            function
                .parameters
                .push(Var::new(self.arena.alloc(Variable {
                    name: Some(format!(
                        "perm_table_{}_{}_{}",
                        perm_table.ident,
                        perm_table.subident_index,
                        perm_table.subident.as_ref().unwrap_or(&"".to_string())
                    )),
                    t: VariableType::PermutationTable,
                })));
        }
        let function = FunctionRef::new(self.arena.alloc(function));
        (function, perm_tables)
    }

    pub fn lower_noise_as_density(
        &mut self,
        density: NormalNoise<'a>,
        permutation_name: &str,
        cname: String,
        origin_scale: (f32, f32, f32),
        position_scale: (f32, f32, f32),
    ) -> DensityFunctionRef<'m> {
        // lower the noise into a density function
        let (function, perm_tables) = lower_normal_noise(
            self.arena,
            density,
            permutation_name,
            cname,
            origin_scale,
            position_scale,
            true,
        );
        let mut density_function = DensityFunction {
            body: function.body,
            canonical_name: function.canonical_name,
            density_inputs: vec![],
            permutation_table_inputs: perm_tables,
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

    pub fn lower_noise_and_mark(
        &mut self,
        noise: NormalNoise<'a>,
        x_scale: f64,
        y_scale: f64,
        z_scale: f64,
        name: String,
    ) -> DensityInput<'m> {
        // check if we already have a density function for this noise
        // create a scaled noise struct to use as a key for the cache
        let noise_scaled = NormalNoiseScaled {
            noise,
            name: name.clone(),
            scale: Scale::new(x_scale as f32, y_scale as f32, z_scale as f32),
        };
        if let Some(cached) = self.noise_inputs.get(&noise_scaled) {
            return cached.clone();
        }

        // borrow the caches from the builder state
        let mut bs = self.builder_state.take().unwrap();
        // lower the noise into a density function
        let density_function_ref = if let Some(cached) = bs.noise_cache.get(&noise_scaled) {
            cached.clone()
        } else {
            let scaled_origin = (
                bs.working_scaled_position.0 * x_scale as f32,
                bs.working_scaled_position.1 * y_scale as f32,
                bs.working_scaled_position.2 * z_scale as f32,
            );
            let scaled_position = (
                bs.working_scaled_position.0 * x_scale as f32,
                bs.working_scaled_position.1 * y_scale as f32,
                bs.working_scaled_position.2 * z_scale as f32,
            );
            let cname = format!("{}_{}", name, bs.noise_cache.len());
            let density_function_ref =
                self.lower_noise_as_density(noise, &name, cname, scaled_origin, scaled_position);
            bs.noise_cache
                .insert(noise_scaled.clone(), density_function_ref.clone());
            density_function_ref
        };
        let v = anonvar(self.arena, VariableType::DensityInput);
        let dimensions = bs.working_dimensions;
        let mut scaled_origin = bs.working_scaled_position;
        scaled_origin.0 *= x_scale as f32;
        scaled_origin.1 *= y_scale as f32;
        scaled_origin.2 *= z_scale as f32;
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
        let scaled_origin = self.builder_state.as_ref().unwrap().working_scaled_position;
        let v = anonvar(self.arena, VariableType::DensityInput);
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
        let old_scaled_origin = bs.working_scaled_position;

        bs.working_dimensions = dimensions;
        bs.working_scaled_position = scaled_origin;
        self.builder_state = Some(bs);

        let density_function_ref = self._lower_density_shader_inner(density, canonical_name);

        let v = anonvar(self.arena, VariableType::DensityInput);
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
        bs.working_scaled_position = old_scaled_origin;
        self.builder_state = Some(bs);
        input
    }

    pub fn lower_density(&mut self, density: Density<'a>) -> Expression<'m> {
        match *density {
            DensityType::Noise {
                ref name,
                noise,
                xz_scale,
                y_scale,
            } => {
                // A noise call specifically
                let fref =
                    self.lower_noise_and_mark(noise, xz_scale, y_scale, xz_scale, name.clone());
                // use the density input as the expression
                Expression::DensityVariable(fref, None)
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
                let xv = anonvar(self.arena, VariableType::F32);
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
                let result_var = anonvar(self.arena, VariableType::F32);
                self.add_variable(result_var.clone());
                self.add_statement(Statement::Assign {
                    target: result_var.clone(),
                    value: result,
                });
                Expression::Variable(result_var)
            }
            DensityType::Interpolated { argument } => {
                // interpolation of the argument,
                // dimensions are quartered
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

                let y_expr = Expression::Field {
                    base: Box::new(Expression::Variable(self.rpos3.clone())),
                    field: "y".to_string(),
                };

                // 2. clamp(normal_y, 0, 1)
                let normal_y = Expression::BinaryOp {
                    op: BinaryOperator::Divide,
                    left: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Subtract,
                        left: Box::new(y_expr),
                        right: Box::new(Expression::Float(from_y)),
                    }),
                    right: Box::new(Expression::Float(to_y - from_y)),
                };

                // 2. clamp(p.y, from_y, to_y)
                let clamped_expr = Expression::ExternCall {
                    function_name: "clamp".to_string(),
                    parameters: vec![normal_y, Expression::Float(0.0), Expression::Float(1.0)],
                    parameter_types: vec![VariableType::F32, VariableType::F32, VariableType::F32],
                };

                // 3. Create clampedY variable
                let clamped_y = anonvar(self.arena, VariableType::F32);
                self.add_variable(clamped_y.clone());

                self.add_statement(Statement::Assign {
                    target: clamped_y.clone(),
                    value: clamped_expr,
                });

                // ---- Build linear mapping ----

                // 4. clampedY * (to_value - from_value) + from_value
                Expression::BinaryOp {
                    op: BinaryOperator::Add,
                    left: Box::new(Expression::BinaryOp {
                        op: BinaryOperator::Multiply,
                        left: Box::new(Expression::Variable(clamped_y.clone())),
                        right: Box::new(Expression::Float(to_value - from_value)),
                    }),
                    right: Box::new(Expression::Float(from_value)),
                }
            }
            DensityType::FlatCache { argument } => {
                // another caching function, for now just ignore the caching and lower the argument
                // TODO: implement caching
                if let Some(cached) = self.density_function_inputs.get(&density) {
                    return Expression::DensityVariable(cached.clone(), None);
                }

                // flat caches are always lowered as separate density functions
                let input = self.lower_density_input(argument, None);
                // add to cache
                self.density_function_inputs.insert(density, input.clone());
                // return the density variable for the input
                Expression::DensityVariable(input, None)
            }
            DensityType::NamedDensityReference { argument, name } => {
                if let Some(cached) = self.density_function_inputs.get(&density) {
                    return Expression::DensityVariable(cached.clone(), None);
                }
                let name = name.clone();
                // flat caches are always lowered as separate density functions
                let input = self.lower_density_input(argument, Some(name));
                // add to cache
                self.density_function_inputs.insert(density, input.clone());
                // return the density variable for the input
                Expression::DensityVariable(input, None)
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
                ref name,
                noise,
                shift_x,
                shift_y,
                shift_z,
                xz_scale,
                y_scale,
            } => {
                // 1. Lower shift densities
                let shift_x_expr = self.lower_density(shift_x);
                let shift_y_expr = self.lower_density(shift_y);
                let shift_z_expr = self.lower_density(shift_z);

                // 2. Build vec3(shiftX, shiftY, shiftZ)
                let shift_vec = Expression::Construct {
                    t: VariableType::Vec3,
                    args: vec![shift_x_expr, shift_y_expr, shift_z_expr],
                };

                // 3. p + shift_vec
                let shifted_position = if xz_scale == 1.0 && y_scale == 1.0 {
                    Expression::BinaryOp {
                        op: BinaryOperator::Add,
                        left: Box::new(Expression::Variable(self.rpos3.clone())),
                        right: Box::new(shift_vec),
                    }
                } else {
                    Expression::BinaryOp {
                        op: BinaryOperator::Add,
                        left: Box::new(Expression::BinaryOp {
                            op: BinaryOperator::Multiply,
                            left: Box::new(Expression::Variable(self.rpos3.clone())),
                            right: Box::new(Expression::Construct {
                                t: VariableType::Vec3,
                                args: vec![
                                    Expression::Float(xz_scale),
                                    Expression::Float(y_scale),
                                    Expression::Float(xz_scale),
                                ],
                            }),
                        }),
                        right: Box::new(shift_vec),
                    }
                };

                let id = self.noise_inputs.len();
                let cname = format!("{}_shifted_{}", name, id);
                let bs = self.builder_state.take().unwrap();

                let scaled_origin = (
                    bs.working_scaled_position.0 * xz_scale as f32,
                    bs.working_scaled_position.1 * y_scale as f32,
                    bs.working_scaled_position.2 * xz_scale as f32,
                );
                let scaled_position = (
                    bs.working_scaled_position.0 * xz_scale as f32,
                    bs.working_scaled_position.1 * y_scale as f32,
                    bs.working_scaled_position.2 * xz_scale as f32,
                );

                self.builder_state = Some(bs);

                // 4. Lower noise but don't mark it, we just want the density function reference
                let (noise_function_ref, perm_tables) =
                    self.lower_noise(noise, scaled_origin, scaled_position, &name, cname);
                // add it as a helper function
                self.helper_functions.push(noise_function_ref.clone());
                let perm_tables_args = perm_tables
                    .clone()
                    .into_iter()
                    .map(|perm_table| Expression::PermutationTable(perm_table));
                // add the permutation tables to the current density function
                self.density_function
                    .permutation_table_inputs
                    .extend(perm_tables);
                let parameters = std::iter::once(shifted_position.into())
                    .chain(perm_tables_args)
                    .collect();
                Expression::FunctionCall {
                    function: noise_function_ref,
                    parameters: parameters,
                }
            }
            DensityType::ShiftA { argument, ref name } => {
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
                let old_dimensions = self.builder_state.as_ref().unwrap().working_dimensions;
                let dimensions = (old_dimensions.0, 1, old_dimensions.2);
                let old_scaled_origin =
                    self.builder_state.as_ref().unwrap().working_scaled_position;
                let scaled_origin = (old_scaled_origin.0 / 4.0, 0.0, old_scaled_origin.2 / 4.0);

                self.builder_state.as_mut().map(|bs| {
                    bs.working_dimensions = dimensions;
                    bs.working_scaled_position = scaled_origin;
                });

                //let density_input =
                //self.lower_noise(argument, None, dimensions, scaled_origin);
                let density_input = self.lower_noise_and_mark(
                    argument,
                    scaled_origin.0 as f64,
                    scaled_origin.1 as f64,
                    scaled_origin.2 as f64,
                    name.clone(),
                );
                let index = Box::new(Expression::ExternCall {
                    function_name: "flat_y_zero_index".into(),
                    parameters: vec![
                        Expression::Variable(self.p.clone()),
                        Expression::Int(old_dimensions.0),
                        Expression::Int(old_dimensions.1),
                    ],
                    parameter_types: vec![VariableType::Pos3, VariableType::I32, VariableType::I32],
                });

                self.builder_state.as_mut().map(|bs| {
                    bs.working_dimensions = old_dimensions;
                    bs.working_scaled_position = old_scaled_origin;
                });
                // let call = Expression::DensityFunctionCall {
                //     function: argument_density_function_ref,
                //     position: shift_vec.into(),
                // };

                let call = Expression::DensityVariable(density_input, Some(index));
                // multiply the result by 4
                Expression::BinaryOp {
                    op: BinaryOperator::Multiply,
                    left: Box::new(call),
                    right: Box::new(Expression::Float(4.0)),
                }
            }
            DensityType::ShiftB { argument, ref name } => {
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

                let old_dimensions = self.builder_state.as_ref().unwrap().working_dimensions;
                let dimensions = (old_dimensions.0, old_dimensions.1, 1);
                let old_scaled_origin =
                    self.builder_state.as_ref().unwrap().working_scaled_position;
                let scaled_origin = (old_scaled_origin.0 / 4.0, old_scaled_origin.1 / 4.0, 0.0);

                self.builder_state.as_mut().map(|bs| {
                    bs.working_dimensions = dimensions;
                    bs.working_scaled_position = scaled_origin;
                });

                //let density_input =
                //self.lower_noise(argument, None, dimensions, scaled_origin);
                let density_input = self.lower_noise_and_mark(
                    argument,
                    scaled_origin.0 as f64,
                    scaled_origin.1 as f64,
                    scaled_origin.2 as f64,
                    name.clone(),
                );

                let index = Box::new(Expression::ExternCall {
                    function_name: "flat_z_zero_index".into(),
                    parameters: vec![
                        Expression::Variable(self.p.clone()),
                        Expression::Int(old_dimensions.0),
                        Expression::Int(old_dimensions.1),
                    ],
                    parameter_types: vec![VariableType::Pos3, VariableType::I32, VariableType::I32],
                });

                self.builder_state.as_mut().map(|bs| {
                    bs.working_dimensions = old_dimensions;
                    bs.working_scaled_position = old_scaled_origin;
                });

                let call = Expression::DensityVariable(density_input, Some(index));

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
                println!(
                    "Lowering Max density with left expr: {:?} and right expr: {:?}",
                    left_expr, right_expr
                );
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

                let v = anonvar(self.arena, VariableType::F32);
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

                let v = anonvar(self.arena, VariableType::F32);
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
                let v = anonvar(self.arena, VariableType::F32);
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
                let v = anonvar(self.arena, VariableType::F32);
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
    p: Var<'m>,
    origin: Var<'m>,
    origin_scale: (f32, f32, f32),
    position_scale: (f32, f32, f32),
) -> Expression<'m> {
    // // rpos3 = origin * origin_scale + p * position_scale
    // if origin_scale == (1.0, 1.0, 1.0) {
    //     return Expression::BinaryOp {
    //         op: BinaryOperator::Add,
    //         left: Box::new(Expression::Variable(origin)),
    //         right: Box::new(Expression::Variable(p)),
    //     };
    // }

    // Expression::BinaryOp {
    //     op: BinaryOperator::Multiply,
    //     left: Box::new(Expression::BinaryOp {
    //         op: BinaryOperator::Add,
    //         left: Box::new(Expression::Variable(origin)),
    //         right: Box::new(Expression::Variable(p)),
    //     }),
    //     right: Box::new(Expression::Construct {
    //         t: VariableType::Vec3,
    //         args: vec![
    //             Expression::Float(scale.0 as f64),
    //             Expression::Float(scale.1 as f64),
    //             Expression::Float(scale.2 as f64),
    //         ],
    //     }),
    // }

    // a few cased:
    // always scaled_origin + scaled_position
    // scaled_origin is always origin * origin_scale
    // scaled_position is always p * position_scale
    // only calculate when the scale is not (1.0, 1.0, 1.0)
    // essentially 4 cases:
    // 1. both scales are (1.0, 1.0, 1.0), then rpos3 = origin + p
    // 2. only origin_scale is not (1.0, 1.0, 1.0), then rpos3 = origin * origin_scale + p
    // 3. only position_scale is not (1.0, 1.0, 1.0), then rpos3 = origin + p * position_scale
    // 4. both scales are not (1.0, 1.0, 1.0), then rpos3 = origin * origin_scale + p * position_scale

    if origin_scale == position_scale {
        // the common case where both scales are the same, we can just do (origin + p) * scale
        if origin_scale == (1.0, 1.0, 1.0) {
            return Expression::BinaryOp {
                op: BinaryOperator::Add,
                left: Box::new(Expression::Variable(origin)),
                right: Box::new(Expression::Variable(p)),
            };
        }
        Expression::BinaryOp {
            op: BinaryOperator::Multiply,
            left: Box::new(Expression::BinaryOp {
                op: BinaryOperator::Add,
                left: Box::new(Expression::Variable(origin)),
                right: Box::new(Expression::Variable(p)),
            }),
            right: Box::new(Expression::Construct {
                t: VariableType::Vec3,
                args: vec![
                    Expression::Float(origin_scale.0 as f64),
                    Expression::Float(origin_scale.1 as f64),
                    Expression::Float(origin_scale.2 as f64),
                ],
            }),
        }
    } else {
        // the more general case where the scales are different, we have to calculate both separately
        let scaled_origin = if origin_scale == (1.0, 1.0, 1.0) {
            Expression::Variable(origin)
        } else {
            Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                left: Box::new(Expression::Variable(origin)),
                right: Box::new(Expression::Construct {
                    t: VariableType::Vec3,
                    args: vec![
                        Expression::Float(origin_scale.0 as f64),
                        Expression::Float(origin_scale.1 as f64),
                        Expression::Float(origin_scale.2 as f64),
                    ],
                }),
            }
        };
        let scaled_position = if position_scale == (1.0, 1.0, 1.0) {
            Expression::Variable(p)
        } else {
            Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                left: Box::new(Expression::Variable(p)),
                right: Box::new(Expression::Construct {
                    t: VariableType::Vec3,
                    args: vec![
                        Expression::Float(position_scale.0 as f64),
                        Expression::Float(position_scale.1 as f64),
                        Expression::Float(position_scale.2 as f64),
                    ],
                }),
            }
        };
        Expression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(scaled_origin),
            right: Box::new(scaled_position),
        }
    }
}
