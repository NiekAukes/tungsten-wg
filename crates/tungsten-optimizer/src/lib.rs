pub mod conversion;
mod fuse;
mod inline;

pub fn perform_inline(program: &mut OptimizedProgram) {
    let inlines = inline::Inliner::new(program).traverse();
    println!("performed {} inlines", inlines);
}
