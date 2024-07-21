use std::error::Error;
#[cfg(test)]
use std::sync::Arc;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

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
