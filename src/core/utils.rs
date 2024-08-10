use std::error::Error;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::core::{EvaluationResult, Evaluator, Individual};

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
