use std::rc::Weak;

use crate::spmt_optimize::model::OptimizedProgram;

mod fuse;
mod inline;
pub mod model;
pub mod conversion;

pub fn perform_inline(program: &mut OptimizedProgram) {
    let inlines = inline::Inliner::new(program).traverse();
    println!("performed {} inlines", inlines);
}