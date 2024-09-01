use crate::algorithms::{NSGA3Arg, NSGA3};
use crate::core::{OError, Problem};

/// Adaptive `NSGA3` algorithm. This is an alias for [`NSGA3`] when the `adaptive` option is set to
/// `true`.
/// This implements the new algorithm from Jain and Deb (2014) to handle problems where not all
/// reference points intersect the optimal Pareto front. This helps to reduce crowding and enhance
/// the solution quality.
///
/// Implemented based on:
/// > Jain, Himanshu & Deb, Kalyanmoy. (2014). An Evolutionary Many-Objective Optimization
/// > Algorithm Using Reference-Point Based Non dominated Sorting Approach, Part II: Handling
/// > Constraints and Extending to an Adaptive Approach. Evolutionary Computation, IEEE
/// > Transactions on. 18. 602-622. <doi.org/10.1109/TEVC.2013.2281534>.
///
/// For a detailed explanation about the implementation see `AdaptiveReferencePoints`.
pub struct AdaptiveNSGA3;

impl AdaptiveNSGA3 {
    /// Initialise the [`NSGA3`] algorithm with `adaptive` option set to `true`.
    ///
    /// returns: `NSGA3`
    pub fn new(problem: Problem, options: NSGA3Arg) -> Result<NSGA3, OError> {
        NSGA3::new(problem, options, true)
    }
}
