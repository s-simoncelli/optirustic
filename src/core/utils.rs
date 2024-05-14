use std::error::Error;

use crate::core::{EvaluationResult, Evaluator, Individual, OError};

/// Return a dummy evaluator. This is only used in tests.
///
/// return `Box<dyn Evaluator>`
#[doc(hidden)]
pub fn dummy_evaluator() -> Box<dyn Evaluator> {
    // dummy evaluator function
    #[derive(Debug)]
    struct UserEvaluator;
    impl Evaluator for UserEvaluator {
        fn evaluate(&self, _: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
            Ok(EvaluationResult {
                constraints: Default::default(),
                objectives: Default::default(),
            })
        }
    }

    Box::new(UserEvaluator)
}

/// Calculate the vector minimum value.
///
/// # Arguments
///
/// * `v`: The vector.
///
/// returns: `Result<f64, OError>`
///
/// # Examples
pub fn vector_min(v: &[f64]) -> Result<f64, OError> {
    Ok(*v
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .ok_or(OError::Generic(
            "Cannot calculate vector min value".to_string(),
        ))?)
}

/// Calculate the vector maximum value.
///
/// # Arguments
///
/// * `v`: The vector.
///
/// returns: `Result<f64, OError>`
///
/// # Examples
pub fn vector_max(v: &[f64]) -> Result<f64, OError> {
    Ok(*v
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(OError::Generic(
            "Cannot calculate vector max value".to_string(),
        ))?)
}

pub fn argsort(data: &[f64]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| data[*a].total_cmp(&data[*b]));
    indices
}
