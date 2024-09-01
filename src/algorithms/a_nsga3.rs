use crate::algorithms::{NSGA3Arg, NSGA3};
use crate::core::{OError, Problem};

/// Adaptive `NSGA3` algorithm. This is an alias for [`NSGA3`] when the `adaptive` option is set to
/// `true`.
pub struct AdaptiveNSGA3;

impl AdaptiveNSGA3 {
    /// Initialise the [`NSGA3`] algorithm with `adaptive` option set to `true`.
    ///
    /// returns: `NSGA3`
    pub fn new(problem: Problem, options: NSGA3Arg) -> Result<NSGA3, OError> {
        NSGA3::new(problem, options, true)
    }
}
