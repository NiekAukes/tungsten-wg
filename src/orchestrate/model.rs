use std::collections::HashMap;

use crate::{
    orchestrate::Scale,
    spmt::model::{Interned, PermutationTableInput},
};

// pub type ShaderRef = Interned<Shader>;
pub type Ref<'a, T> = Interned<'a, T>;
pub type ShaderRef<'a> = Ref<'a, Shader<'a>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderDependency<'a> {
    pub shader: ShaderRef<'a>,
    pub scaled_origin: Scale,
    pub dimensions: (i32, i32, i32),
}

pub struct Shader<'a> {
    pub name: String,
    pub source: String,
    pub inputs: Vec<ShaderDependency<'a>>,
    pub permutation_tables: Vec<PermutationTableInput>,
}

pub struct Orchestration<'a> {
    pub shaders: Vec<ShaderRef<'a>>,
    pub arena: &'a bumpalo::Bump,
    pub main_shaders: Vec<ShaderDependency<'a>>,
}

pub trait Interner<'a, T: Sized> {
    fn intern(&mut self, shader: T) -> Ref<'a, T>;
}

impl<'a, T: Sized> Interner<'a, T> for Orchestration<'a> {
    fn intern(&mut self, shader: T) -> Ref<'a, T> {
        Ref::new(self.arena.alloc(shader))
    }
}

impl<'a> Orchestration<'a> {
    pub fn new(arena: &'a bumpalo::Bump) -> Self {
        Self {
            shaders: Vec::new(),
            arena,
            main_shaders: Vec::new(),
        }
    }

    pub fn add_shader(&mut self, shader: Shader<'a>) -> ShaderRef<'a> {
        let shader_ref = self.intern(shader);
        self.shaders.push(shader_ref);
        shader_ref
    }

    pub fn add_main_shader(
        &mut self,
        shader: Shader<'a>,
        dimensions: (i32, i32, i32),
    ) -> ShaderRef<'a> {
        let shader_ref = self.add_shader(shader);
        self.main_shaders.push(ShaderDependency {
            shader: shader_ref,
            scaled_origin: Scale::new(1.0, 1.0, 1.0),
            dimensions,
        });
        shader_ref
    }
}

impl PartialOrd for ShaderDependency<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dimensions.partial_cmp(&other.dimensions)
    }
}

impl Ord for ShaderDependency<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dimensions.cmp(&other.dimensions)
    }
}
