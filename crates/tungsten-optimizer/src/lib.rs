mod fuse;
mod inline;

use tungsten_ir::model::OptimizedProgram;

/// What an inlining run changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InlineStats {
    /// Number of compute units that were inlined into their caller.
    pub inlined: usize,
}

pub fn perform_inline(program: &mut OptimizedProgram) -> InlineStats {
    let inlined = inline::Inliner::new(program).traverse();
    InlineStats { inlined }
}
