use std::{ops::Deref, rc::Rc};

use crate::{
    parse::model::{Density, DensityType, NormalNoise},
    spmt::model::{DensityFunction, DensityFunctionRef, Expression, SPMT, Variable, VariableType},
    transform_spmt::density::DensityBuilder,
};

pub mod density;
pub mod noise;
pub mod spline;

fn lit(v: f64) -> Expression<'static> {
    Expression::Float(v)
}

pub fn newvar(name: &str, t: VariableType) -> Rc<Variable> {
    Rc::new(Variable {
        name: Some(name.into()),
        t,
    })
}

pub fn anonvar(t: VariableType) -> Rc<Variable> {
    Rc::new(Variable { name: None, t })
}

type DensityFunctionCache<'a, 'm> = std::collections::HashMap<Density<'a>, DensityFunctionRef<'m>>;
type NoiseCache<'a, 'm> = std::collections::HashMap<NormalNoise<'a>, DensityFunctionRef<'m>>;

pub struct BuilderState<'a, 'm> {
    pub density_function_cache: DensityFunctionCache<'a, 'm>,
    pub noise_cache: NoiseCache<'a, 'm>,

    working_dimensions: (i32, i32, i32),
    working_scaled_origin: (f32, f32, f32),
}

pub struct Transformer<'a, 'm> {
    pub final_model: SPMT<'m>,
    pub arena: &'m bumpalo::Bump,
    pub builder_state: Option<BuilderState<'a, 'm>>,
}

impl<'a, 'm> Transformer<'a, 'm> {
    pub fn new(arena: &'m bumpalo::Bump, initial_working_dimensions: (i32, i32, i32)) -> Self {
        Self {
            final_model: SPMT {
                density_functions: Vec::new(),
                functions: Vec::new(),
                main_density_functions: Vec::new(),
            },
            arena,
            // density_function_cache: Option::Some(std::collections::HashMap::new()),
            // noise_cache: Option::Some(std::collections::HashMap::new()),
            builder_state: Option::Some(BuilderState {
                density_function_cache: std::collections::HashMap::new(),
                noise_cache: std::collections::HashMap::new(),
                working_dimensions: initial_working_dimensions,
                working_scaled_origin: (1.0, 1.0, 1.0),
            }),
        }
    }

    pub fn transform(
        mut self,
        noise_generator: &'a crate::parse::NoiseGeneratorSettings<'a>,
    ) -> SPMT<'m> {
        // For each density function in the Minecraft data, lower it and add it to the final model
        for density in noise_generator.noise_router.all_densities() {
            let density_function = self.lower_density_function(density);
            //self.final_model.density_functions.push(density_function);
            self.final_model
                .main_density_functions
                .push(density_function);
            self.final_model.density_functions.push(density_function);
        }

        let BuilderState {
            density_function_cache,
            noise_cache,
            ..
        } = self.builder_state.take().unwrap();

        println!(
            "Density function cache size: {}, Noise cache size: {}",
            density_function_cache.len(),
            noise_cache.len()
        );

        for density_function in density_function_cache.values() {
            // check if the density function is already in the main density functions, and if not, add it to the final model
            if !self
                .final_model
                .main_density_functions
                .iter()
                .any(|df| df.canonical_name == density_function.canonical_name)
             {
                self.final_model.density_functions.push(*density_function);
            } 
            //self.final_model.density_functions.push(*density_function);
        }
        for noise in noise_cache.values() {
            self.final_model.density_functions.push(*noise);
        }

        self.final_model
    }

    pub fn lower_density_function(&mut self, mut density: Density<'a>) -> DensityFunctionRef<'m> {
        let bs = self.builder_state.take().unwrap();
        if let Some(cached) = bs.density_function_cache.get(&density) {
            let ret = cached.clone();
            self.builder_state = Some(bs);
            return ret;
        }

        let mut name = None;
        if let DensityType::NamedDensityReference {
            name: dname,
            argument,
        } = *density
        {
            density = argument;
            name = Some(dname.clone())
        }

        // create a density function builder
        let mut builder = DensityBuilder::new_named(self.arena, bs, name);
        // lower the density into the builder
        let r = builder.lower_density(density);
        // build the density function
        let (density_function, helpers, mut bs) = builder.finish(r);
        self.final_model.functions.extend(helpers);
        let density_function: DensityFunctionRef<'m> =
            DensityFunctionRef::new(self.arena.alloc(density_function));
        let _a = density_function.canonical_name.as_ref().unwrap();

        bs.density_function_cache
            .insert(density, density_function.clone());
        self.builder_state = Some(bs);
        density_function
    }
}
