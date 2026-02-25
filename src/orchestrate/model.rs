use std::collections::HashMap;

use crate::spmt::model::Interned;

// pub type ShaderRef = Interned<Shader>;
pub type Ref<'a, T> = Interned<'a, T>;
pub type ShaderRef<'a> = Ref<'a, Shader<'a>>;

pub struct Shader<'a> {
    pub name: String,
    pub source: String,
    pub inputs: Vec<ShaderRef<'a>>,
}

pub struct Orchestration<'a> {
    pub shaders: Vec<ShaderRef<'a>>,
    pub arena: &'a bumpalo::Bump,
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
        }
    }

    pub fn add_shader(&mut self, shader: Shader<'a>) -> ShaderRef<'a> {
        let shader_ref = self.intern(shader);
        self.shaders.push(shader_ref);
        shader_ref
    }
}
