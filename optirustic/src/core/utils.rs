use std::error::Error;
#[cfg(test)]
use std::sync::Arc;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[cfg(test)]
use crate::core::{BoundedNumber, Objective, ObjectiveDirection, Problem, VariableType};
use crate::core::{EvaluationResult, Evaluator, Individual};
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
        objectives.push(Objective::new(format!("obj{i}").as_str(), *direction));
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
