use std::collections::HashMap;
use std::rc::Rc;

use crate::{
    orchestrate::model::{ShaderDependency, ShaderRef},
    rcl::{Expression, Struct, Variable},
    transform_rcl::sanitize_name,
};

pub const OUTPUT_STRUCT_NAME: &str = "OrchestrationOutput";

/// Builds the `OrchestrationOutput` struct definition and the matching field
/// initialiser list for the final `return` expression.
pub fn build_return_struct<'m>(
    returns: &[ShaderDependency<'m>],
    shader_output_map: &HashMap<ShaderRef<'m>, Rc<Variable>>,
) -> (Struct, Vec<(String, Expression<'m>)>) {
    let mut output_struct = Struct::new(OUTPUT_STRUCT_NAME.to_string());
    let mut struct_fields = Vec::new();

    for dep in returns {
        let output_var = shader_output_map.get(&dep.shader).unwrap();
        let field_name = sanitize_name(&dep.shader.name);
        output_struct.add_field(field_name.clone(), output_var.t.clone());
        struct_fields.push((field_name, Expression::Variable(output_var.clone())));
    }

    (output_struct, struct_fields)
}
