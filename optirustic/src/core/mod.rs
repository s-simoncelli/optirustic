pub use constraint::{Constraint, RelationalOperator};
pub use error::OError;
pub use individual::{Individual, IndividualExport, Individuals, IndividualsMut, Population};
pub use objective::{Objective, ObjectiveDirection};
pub use problem::{builtin_problems, EvaluationResult, Evaluator, Problem, ProblemExport};
pub use variable::{Boolean, BoundedNumber, Choice, Variable, VariableType, VariableValue};

mod constraint;
mod error;
mod individual;
mod objective;
mod problem;
pub(crate) mod test_utils;
pub mod utils;
mod variable;
