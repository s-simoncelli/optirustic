use std::error::Error;
use std::ops::Range;

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
pub fn vector_max(v: &[f64]) -> Result<f64, OError> {
    Ok(*v
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(OError::Generic(
            "Cannot calculate vector max value".to_string(),
        ))?)
}

/// Returns the indices that would sort an array.
///
/// # Arguments
///
/// * `data`: The vector to sort.
///
/// returns: `Vec<usize>`. The vector with the indices.
pub fn argsort(data: &[f64]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| data[*a].total_cmp(&data[*b]));
    indices
}

/// Get the vector values outside a lower and upper bounds.
///
/// # Arguments
///
/// * `vector`: The vector.
/// * `range`: The range.
///
/// returns: `Vec<f64>` The values outside the range.
#[cfg(test)]
pub(crate) fn check_value_in_range(vector: &[f64], range: &Range<f64>) -> Vec<f64> {
    vector
        .iter()
        .filter_map(|v| if !range.contains(v) { Some(*v) } else { None })
        .collect()
}

/// Check if a number matches another one, but using ranges. Return the vector items outside
/// `strict_range`, if their number is above `max_outside_strict_range`; otherwise the items outside
/// items `loose_range`. This is used to check whether a value from a genetic algorithm matches an
/// exact value; sometimes an algorithm gets very close to the expected value but the solution
/// is still acceptable within a tolerance.
///
/// # Arguments
///
/// * `vector`: The vector.
/// * `strict_range`: The strict range.
/// * `loose_range`: The loose bound.
/// * `max_outside_strict_range`: The maximum item numbers that can be outside the `strict_range`.
///
/// returns: `(Vec<f64>, Range<f64>)` The values outside the range in the tuple second item.
#[cfg(test)]
pub(crate) fn check_exact_value(
    vector: &[f64],
    strict_range: &Range<f64>,
    loose_range: &Range<f64>,
    max_outside_strict_range: usize,
) -> (Vec<f64>, Range<f64>, String) {
    let v_outside = check_value_in_range(vector, strict_range);

    if v_outside.len() > max_outside_strict_range {
        (v_outside, strict_range.clone(), "strict".to_string())
    } else {
        let v_loose_outside = check_value_in_range(&v_outside, loose_range);
        (v_loose_outside, loose_range.clone(), "loose".to_string())
    }
}
