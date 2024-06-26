use thiserror::Error;

#[derive(Error, Debug)]
/// Errors raised by the library.
pub enum OError {
    #[error("The following error occurred: {0}")]
    Generic(String),
    #[error("You must provide at least one objective to properly define a problem")]
    NoObjective,
    #[error("You must provide at least one variable to properly define a problem")]
    NoVariables,
    #[error("The {0} type named '{1}' already exist")]
    DuplicatedName(String, String),
    #[error("The {0} index {1} does not exist")]
    NonExistingIndex(String, usize),
    #[error("The {0} named '{1}' does not exist")]
    NonExistingName(String, String),
    #[error("The variable type set on the problem '{0}' does not match the provided value")]
    NonMatchingVariableType(String),
    #[error("The variable is not {0}")]
    WrongVariableType(String),
    #[error("The variable '{0}' is not {1}")]
    WrongVariableTypeWithName(String, String),
    #[error("The min value ({0}) must be strictly smaller than the max value ({1}).")]
    TooLargeLowerBound(String, String),
    #[error("The data named {0} is not set on the individual")]
    WrongDataName(String),
    #[error("The data type is not {0}")]
    WrongDataType(String),
    #[error("An error occurred in the comparison operator '{0}': {1}")]
    ComparisonOperator(String, String),
    #[error("An error occurred in the selector operator '{0}': {1}")]
    SelectorOperator(String, String),
    #[error("An error occurred in the crossover operator '{0}': {1}")]
    CrossoverOperator(String, String),
    #[error("An error occurred in the mutation operator '{0}': {1}")]
    MutationOperator(String, String),
    #[error("An error occurred in the survival operator '{0}': {1}")]
    SurvivalOperator(String, String),
    #[error("An error occurred when evaluating a solution: {0}")]
    Evaluation(String),
    #[error("An error occurred in the calculation of the '{0}' metric: {1}")]
    Metric(String, String),
    #[error("An error occurred when initialising {0}: {1}")]
    AlgorithmInit(String, String),
    #[error("An error occurred when running {0}: {1}")]
    AlgorithmRun(String, String),
    #[error("An error occurred when exporting the algorithm data: {0}")]
    AlgorithmExport(String),
    #[error("NaN detected when adding {0} '{1}'. This may be an error in the user-defined evaluation function")]
    NaN(String, String),
}
