use std::error::Error;
#[cfg(test)]
use std::ops::Range;
#[cfg(test)]
use std::sync::Arc;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[cfg(test)]
use crate::core::{BoundedNumber, Objective, ObjectiveDirection, Problem, VariableType};
use crate::core::{EvaluationResult, Evaluator, Individual, OError};
#[cfg(test)]
use crate::core::problem::builtin_problems::ztd1;

/// Get the random number generator. If no seed is provided, this randomly generated.
///
/// # Arguments
///
/// * `seed`: The optional seed number.
///
/// returns: `Box<dyn RngCore>`
pub(crate) fn get_rng(seed: Option<u64>) -> Box<dyn RngCore> {
    let rng = match seed {
        None => ChaCha8Rng::from_seed(Default::default()),
        Some(s) => ChaCha8Rng::seed_from_u64(s),
    };
    Box::new(rng)
}

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

/// Define the sort type
#[derive(PartialEq)]
pub enum Sort {
    /// Sort values in ascending order
    Ascending,
    /// Sort values in descending order
    Descending,
}

/// Returns the indices that would sort an array in ascending order.
///
/// # Arguments
///
/// * `data`: The vector to sort.
/// * `sort_type`: Specify whether to sort in ascending or descending order.
///
/// returns: `Vec<usize>`. The vector with the indices.
pub fn argsort(data: &[f64], sort_type: Sort) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| data[*a].total_cmp(&data[*b]));

    if sort_type == Sort::Descending {
        indices.reverse();
    }
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
    if strict_range == loose_range {
        panic!("Bounds are identical");
    }
    let v_outside = check_value_in_range(vector, strict_range);

    if v_outside.len() > max_outside_strict_range {
        (v_outside, strict_range.clone(), "strict".to_string())
    } else {
        let v_loose_outside = check_value_in_range(&v_outside, loose_range);
        (v_loose_outside, loose_range.clone(), "loose".to_string())
    }
}

/// Create the individuals for a `N`-objective dummy problem, where `N` is the number of items in
/// the arrays of `objective_values`.
///
/// # Arguments
///
/// * `objective_values`: The objective values to set on the individuals. A number of individuals
/// equal to this vector size will be created.
/// * `objective_direction`: The direction of each objective.
///
/// returns: `Vec<Individual>`
#[cfg(test)]
pub(crate) fn individuals_from_obj_values_dummy<const N: usize>(
    objective_values: &Vec<[f64; N]>,
    objective_direction: &[ObjectiveDirection; N],
) -> Vec<Individual> {
    let mut objectives = Vec::new();
    for (i, direction) in objective_direction.iter().enumerate() {
        objectives.push(Objective::new(
            format!("obj{i}").as_str(),
            direction.clone(),
        ));
    }
    let variables = vec![VariableType::Real(
        BoundedNumber::new("X", 0.0, 2.0).unwrap(),
    )];
    let problem = Arc::new(Problem::new(objectives, variables, None, dummy_evaluator()).unwrap());

    // create the individuals
    let mut individuals: Vec<Individual> = Vec::new();
    for data in objective_values {
        let mut individual = Individual::new(problem.clone());
        for (oi, obj_value) in data.iter().enumerate() {
            individual
                .update_objective(format!("obj{oi}").as_str(), *obj_value)
                .unwrap();
        }
        individuals.push(individual);
    }

    individuals
}

/// Build the vectors with the individuals and assign the objective values for a ZTD1 problem
///
/// # Arguments
///
/// * `obj_values`: The objective to use. The size of this vector corresponds to the population
///  size and the size of the nested vector to the number of problem objectives.
///
/// returns: `Vec<Individual>`
#[cfg(test)]
pub(crate) fn individuals_from_obj_values_ztd1(obj_values: &[Vec<f64>]) -> Vec<Individual> {
    let problem = Arc::new(ztd1(obj_values.len()).unwrap());
    let mut individuals = vec![];
    for value in obj_values {
        let mut i = Individual::new(problem.clone());
        i.update_objective("f1", value[0]).unwrap();
        i.update_objective("f2", value[1]).unwrap();
        individuals.push(i);
    }
    individuals
}

#[cfg(test)]
mod test {
    use crate::core::utils::{argsort, Sort};

    #[test]
    fn test_argsort() {
        let vec = vec![99.0, 11.0, 456.2, 19.0, 0.5];

        assert_eq!(argsort(&vec, Sort::Ascending), vec![4, 1, 3, 0, 2]);
        assert_eq!(argsort(&vec, Sort::Descending), vec![2, 0, 3, 1, 4]);
    }
}
