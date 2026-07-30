use std::cmp::Ordering;

use crate::{
    algorithms::CROWDING_DIST_KEY,
    core::{Individual, OError},
    utils::{get_pareto_constrained_dominance, PreferredSolution, RANK_KEY},
};

/// A trait to implement a comparison operator between two solutions.
pub trait BinaryComparisonOperator {
    /// Compare two solution and select the best one.
    ///
    /// # Arguments
    ///
    /// * `first_solution`: The first solution to compare.
    /// * `second_solution`: The second solution to compare.
    ///
    /// returns: `Result<PreferredSolution, OError>` The preferred solution.
    fn compare(
        first_solution: &Individual,
        second_solution: &Individual,
    ) -> Result<PreferredSolution, OError>
    where
        Self: Sized;
}

/// This assesses the Pareto dominance between two solutions $S_1$ and $S_2$ and their constraint
/// violations in constrained multi-objective optimization problems. A solution $S_1$ is
/// constraint-dominated if:
/// 1) $S_1$ is feasible but $S_2$ is not.
/// 2) Both $S_1$ and $S_2$ are infeasible and $CV(S_1) < CV(S_2)$ (where $CV$ is the constraint
///    violation function); or
/// 3) both are feasible and $S_1$ Pareto-dominate $S_2$ ($ S_1 \prec S_2 $).
///
///
/// See:
///  - Kalyanmoy Deb & Samir Agrawal. (2002). <https://doi.org/10.1007/978-3-7091-6384-9_40>.
///  - Shuang Li, Ke Li, Wei Li. (2022). <https://doi.org/10.48550/arXiv.2205.14349>.
///
pub struct ParetoConstrainedDominance;

impl BinaryComparisonOperator for ParetoConstrainedDominance {
    /// Get the dominance relation between two solutions with constraints.
    ///
    /// # Arguments
    ///
    /// * `first_solution`: The first solution to compare.
    /// * `second_solution`: The second solution to compare.
    ///
    /// returns: `Result<PreferredSolution, OError>` The dominance relation between solution 1
    /// and 2.
    fn compare(
        first_solution: &Individual,
        second_solution: &Individual,
    ) -> Result<PreferredSolution, OError> {
        get_pareto_constrained_dominance(first_solution, second_solution, None)
    }
}

/// This implements the crowded-comparison operator from Deb et al. (2002) for the NSGAII algorithm.
/// A solution $S_i$ dominates a solution $S_j$ if:
///
///    - $rank_i < rank_j$
///
/// or when $rank_i =rank_j$
///
///    - ${distance}_i > {distance}_j$
///
/// where $rank_x$ is the rank from the fast non-dominated sort algorithm (see
/// [`crate::utils::fast_non_dominated_sort()`]) and $distance_x$ is the crowding distance using
/// neighboring solutions.
///
/// Implemented based on:
/// > K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multi-objective genetic
/// > algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp.
/// > 182-197, April 2002, doi: 10.1109/4235.996017.
///
pub struct CrowdedComparison;

impl BinaryComparisonOperator for CrowdedComparison {
    /// Get the crowded comparison relation between two solutions with rank and crowding distance
    /// data. This returns an error if the data does not exist on either solutions.
    ///
    /// # Arguments
    ///
    /// * `first_solution`: The first solution to compare.
    /// * `second_solution`: The second solution to compare.
    ///
    /// returns: `Result<PreferredSolution, OError>` The dominance relation between solution 1
    /// and 2.
    fn compare(
        first_solution: &Individual,
        second_solution: &Individual,
    ) -> Result<PreferredSolution, OError> {
        let name = "CrowdedComparison".to_string();
        let rank1 = match first_solution.get_data(RANK_KEY) {
            Err(_) => {
                return Err(OError::ComparisonOperator(
                    name,
                    "The rank on the first individual does not exist".to_string(),
                ))
            }
            Ok(r) => r.as_integer()?,
        };
        let rank2 = match second_solution.get_data(RANK_KEY) {
            Err(_) => {
                return Err(OError::ComparisonOperator(
                    name,
                    "The rank on the second individual does not exist".to_string(),
                ))
            }
            Ok(r) => r.as_integer()?,
        };

        match rank1.cmp(&rank2) {
            Ordering::Less => Ok(PreferredSolution::First),
            Ordering::Equal => {
                let d1 = match first_solution.get_data(CROWDING_DIST_KEY) {
                    Err(_) => {
                        return Err(OError::ComparisonOperator(
                            name,
                            format!(
                                "The crowding distance on the first individual {:?} does not exist",
                                first_solution.variables()
                            ),
                        ))
                    }
                    Ok(r) => r.as_real()?,
                };
                let d2 = match second_solution.get_data(CROWDING_DIST_KEY) {
                    Err(_) => {
                        return Err(OError::ComparisonOperator(
                            name,
                            format!(
                            "The crowding distance on the second individual {:?} does not exist",
                            second_solution.variables()
                        ),
                        ))
                    }
                    Ok(r) => r.as_real()?,
                };

                if d1 > d2 {
                    Ok(PreferredSolution::First)
                } else {
                    Ok(PreferredSolution::Second)
                }
            }
            Ordering::Greater => Ok(PreferredSolution::Second),
        }
    }
}

#[cfg(test)]
mod test_crowded_comparison {
    use std::sync::Arc;

    use crate::algorithms::CROWDING_DIST_KEY;
    use crate::core::utils::dummy_evaluator;
    use crate::core::{
        BoundedNumber, DataValue, Individual, Objective, ObjectiveDirection, Problem, VariableType,
    };
    use crate::operators::comparison::CrowdedComparison;
    use crate::operators::BinaryComparisonOperator;
    use crate::utils::{PreferredSolution, RANK_KEY};

    #[test]
    fn test_different_rank() {
        let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let e = dummy_evaluator();
        let problem = Arc::new(Problem::new(objectives, variables, None, e).unwrap());

        let mut solution1 = Individual::new(problem.clone());
        let mut solution2 = Individual::new(problem.clone());
        solution1.set_data(RANK_KEY, DataValue::Integer(1));
        solution2.set_data(RANK_KEY, DataValue::Integer(4));

        // Sol 1 dominates
        assert_eq!(
            CrowdedComparison::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::First
        );

        // Sol 2 dominates
        solution1.set_data(RANK_KEY, DataValue::Integer(5));
        assert_eq!(
            CrowdedComparison::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::Second
        );
    }

    #[test]
    fn test_same_rank() {
        let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let e = dummy_evaluator();
        let problem = Arc::new(Problem::new(objectives, variables, None, e).unwrap());

        let mut solution1 = Individual::new(problem.clone());
        let mut solution2 = Individual::new(problem.clone());
        solution1.set_data(RANK_KEY, DataValue::Integer(1));
        solution2.set_data(RANK_KEY, DataValue::Integer(1));

        solution1.set_data(CROWDING_DIST_KEY, DataValue::Real(10.5));
        solution2.set_data(CROWDING_DIST_KEY, DataValue::Real(0.32));
        // Sol 1 dominates
        assert_eq!(
            CrowdedComparison::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::First
        );

        // Sol 2 dominates
        solution2.set_data(CROWDING_DIST_KEY, DataValue::Real(100.32));
        assert_eq!(
            CrowdedComparison::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::Second
        );
    }
}
