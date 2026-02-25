use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "./tungsten_parse/tungsten.pest"]
pub struct TungstenParser;

pub fn parse_tungsten(
    input: &str,
) -> Result<pest::iterators::Pairs<Rule>, pest::error::Error<Rule>> {
    TungstenParser::parse(Rule::file, input)
}
