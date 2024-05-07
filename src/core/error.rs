use thiserror::Error;

#[derive(Error, Debug)]
pub enum OError {
    #[error("You must provide at least one objective to properly define a problem")]
    NoObjective,
    #[error("You must provide at least one variable to properly define a problem")]
    NoVariables,
    #[error("The {0} type named '{1}' already exist")]
    DuplicatedName(String, String),
    #[error("The {0} named '{1}' does not exist")]
    NonExistingName(String, String),
    #[error("The variable type set on the problem '{0}' does not match the provided value")]
    NonMatchingVariableType(String),
    #[error("The min value ({0}) must be strictly smaller than the max value ({1}).")]
    TooLargeLowerBound(String, String),
}
