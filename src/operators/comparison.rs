use crate::core::{Individual, OError};

/// The preferred solution with the `BinaryComparisonOperator`.
#[derive(Debug, PartialOrd, PartialEq)]
pub enum PreferredSolution {
    /// The first solution is preferred.
    First,
    /// The second solution is preferred.
    Second,
    /// The two solutions are mutually preferred.
    MutuallyPreferred,
}

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
/// violation function); or
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
        let problem = first_solution.problem();
        let cv1 = first_solution.constraint_violation();
        let cv2 = second_solution.constraint_violation();

        // at least one solution is not feasible (step 1-2)
        if problem.number_of_constraints() > 0 && cv1 != cv2 {
            if first_solution.is_feasible() {
                // solution 1 dominates
                return Ok(PreferredSolution::First);
            } else if second_solution.is_feasible() {
                // solution 2 dominates
                return Ok(PreferredSolution::Second);
            } else if cv1 < cv2 {
                // solution 1 dominates
                return Ok(PreferredSolution::First);
            } else if cv1 > cv2 {
                // solution 2 dominates
                return Ok(PreferredSolution::Second);
            }
        }

        // check pareto dominance using all the objectives (step 2)
        let mut relation = PreferredSolution::MutuallyPreferred;
        for objective_name in problem.objective_names() {
            let mut obj_sol1 = first_solution.get_objective_value(objective_name.as_str())?;
            let mut obj_sol2 = second_solution.get_objective_value(objective_name.as_str())?;

            // invert objective value for maximisation problems
            if !problem.is_objective_minimised(objective_name.as_str())? {
                obj_sol1 = -obj_sol1;
                obj_sol2 = -obj_sol2;
            }
            if obj_sol1 < obj_sol2 {
                if relation == PreferredSolution::Second {
                    // mutually dominated
                    return Ok(PreferredSolution::MutuallyPreferred);
                }
                relation = PreferredSolution::First;
            } else if obj_sol1 > obj_sol2 {
                if relation == PreferredSolution::First {
                    // mutually dominated
                    return Ok(PreferredSolution::MutuallyPreferred);
                }
                relation = PreferredSolution::Second;
            }
        }

        Ok(relation)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::core::{
        BoundedNumber, Constraint, Individual, Objective, ObjectiveDirection, Problem,
        RelationalOperator, VariableType,
    };
    use crate::core::utils::dummy_evaluator;
    use crate::operators::{
        BinaryComparisonOperator, ParetoConstrainedDominance, PreferredSolution,
    };

    #[test]
    /// Test unconstrained problem with one objective
    fn test_unconstrained_solutions_1_objective() {
        let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let e = dummy_evaluator();
        let problem = Arc::new(Problem::new(objectives, variables, None, e).unwrap());

        let mut solution1 = Individual::new(problem.clone());
        let mut solution2 = Individual::new(problem.clone());

        // Sol 1 dominates
        solution1.update_objective("obj1", 5.0).unwrap();
        solution2.update_objective("obj1", 15.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::First
        );

        // Sol 2 dominates
        solution1.update_objective("obj1", 5.0).unwrap();
        solution2.update_objective("obj1", 1.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::Second
        );

        // Both are non dominated
        solution1.update_objective("obj1", 5.0).unwrap();
        solution2.update_objective("obj1", 5.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::MutuallyPreferred
        );

        // Maximisation problem
        let objectives = vec![Objective::new("obj1", ObjectiveDirection::Maximise)];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let e = dummy_evaluator();
        let problem = Arc::new(Problem::new(objectives, variables, None, e).unwrap());

        let mut solution1 = Individual::new(problem.clone());
        let mut solution2 = Individual::new(problem.clone());

        // Sol 2 dominates with largets objective
        solution1.update_objective("obj1", 5.0).unwrap();
        solution2.update_objective("obj1", 15.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::Second
        );
    }

    #[test]
    /// Test unconstrained problem with two objectives
    fn test_unconstrained_solutions_2_objectives() {
        let objectives = vec![
            Objective::new("obj1", ObjectiveDirection::Minimise),
            Objective::new("obj2", ObjectiveDirection::Minimise),
        ];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let e = dummy_evaluator();
        let problem = Arc::new(Problem::new(objectives, variables, None, e).unwrap());

        let mut solution1 = Individual::new(problem.clone());
        let mut solution2 = Individual::new(problem.clone());

        // Sol 1 dominates
        solution1.update_objective("obj1", 5.0).unwrap();
        solution1.update_objective("obj2", 1.0).unwrap();
        solution2.update_objective("obj1", 15.0).unwrap();
        solution2.update_objective("obj2", 25.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::First
        );

        // Sol 2 dominates
        solution1.update_objective("obj1", 5.0).unwrap();
        solution1.update_objective("obj2", 1.0).unwrap();
        solution2.update_objective("obj1", -15.0).unwrap();
        solution2.update_objective("obj2", -25.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::Second
        );

        // Obj1 of Sol 1 dominates and Obj2 of Sol 2 dominates
        solution1.update_objective("obj1", 5.0).unwrap();
        solution1.update_objective("obj2", 100.0).unwrap();
        solution2.update_objective("obj1", 15.0).unwrap();
        solution2.update_objective("obj2", 25.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::MutuallyPreferred
        );

        // compare three solutions
        let mut solution3 = Individual::new(problem.clone());
        solution1.update_objective("obj1", 0.0).unwrap();
        solution1.update_objective("obj2", 0.0).unwrap();
        solution2.update_objective("obj1", 1.0).unwrap();
        solution2.update_objective("obj2", 1.0).unwrap();
        solution3.update_objective("obj1", 0.0).unwrap();
        solution3.update_objective("obj2", 1.0).unwrap();

        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::First
        );
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution2, &solution1).unwrap(),
            PreferredSolution::Second
        );
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution3).unwrap(),
            PreferredSolution::First
        );
        // mutually dominated for obj1, but obj2 of sol1 dominates
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution3, &solution1).unwrap(),
            PreferredSolution::Second
        );

        // non-dominance
        solution1.update_objective("obj1", 0.0).unwrap();
        solution1.update_objective("obj2", 1.0).unwrap();
        solution2.update_objective("obj1", 0.5).unwrap();
        solution2.update_objective("obj2", 0.5).unwrap();
        solution3.update_objective("obj1", 1.0).unwrap();
        solution3.update_objective("obj2", 0.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::MutuallyPreferred
        );
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution2, &solution1).unwrap(),
            PreferredSolution::MutuallyPreferred
        );
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution2, &solution3).unwrap(),
            PreferredSolution::MutuallyPreferred
        );
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution3, &solution2).unwrap(),
            PreferredSolution::MutuallyPreferred
        );
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution3).unwrap(),
            PreferredSolution::MutuallyPreferred
        );
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution3, &solution1).unwrap(),
            PreferredSolution::MutuallyPreferred
        );

        // Maximisation problem
        let objectives = vec![
            Objective::new("obj1", ObjectiveDirection::Minimise),
            Objective::new("obj2", ObjectiveDirection::Maximise),
        ];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let e = dummy_evaluator();
        let problem = Arc::new(Problem::new(objectives, variables, None, e).unwrap());

        let mut solution1 = Individual::new(problem.clone());
        let mut solution2 = Individual::new(problem.clone());

        // Neither dominates
        solution1.update_objective("obj1", 5.0).unwrap();
        solution2.update_objective("obj1", 15.0).unwrap();
        solution1.update_objective("obj2", 5.0).unwrap();
        solution2.update_objective("obj2", 15.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::MutuallyPreferred
        );

        // Sol 2 dominates
        solution1.update_objective("obj1", 5.0).unwrap();
        solution1.update_objective("obj2", -5.0).unwrap();
        solution2.update_objective("obj1", 1.0).unwrap();
        solution2.update_objective("obj2", 15.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::Second
        );
    }

    #[test]
    /// Test constrained problem with. The constraint violation determines the dominance relation.
    fn test_constrained_solutions() {
        let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let constraints = vec![Constraint::new("c1", RelationalOperator::EqualTo, 1.0)];

        let e = dummy_evaluator();
        let problem = Arc::new(Problem::new(objectives, variables, Some(constraints), e).unwrap());

        let mut solution1 = Individual::new(problem.clone());
        let mut solution2 = Individual::new(problem.clone());
        solution1.update_objective("obj1", 5.0).unwrap();
        solution2.update_objective("obj1", 15.0).unwrap();

        // Sol 2 dominates because is feasible
        solution1.update_constraint("c1", 0.0).unwrap();
        solution2.update_constraint("c1", 1.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::Second
        );

        // Solution 1 dominates due to the smaller violation
        solution1.update_constraint("c1", 0.5).unwrap();
        solution2.update_constraint("c1", 3.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::First
        );

        // Solution 1 is returned when violation magnitude is the same
        solution1.update_constraint("c1", 0.5).unwrap();
        solution2.update_constraint("c1", 0.5).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::First
        );

        // Two objectives
        let objectives = vec![
            Objective::new("obj1", ObjectiveDirection::Minimise),
            Objective::new("obj2", ObjectiveDirection::Minimise),
        ];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let constraints = vec![Constraint::new("c1", RelationalOperator::EqualTo, 5.0)];

        let e = dummy_evaluator();
        let problem2 = Arc::new(Problem::new(objectives, variables, Some(constraints), e).unwrap());

        let mut solution1 = Individual::new(problem2.clone());
        let mut solution2 = Individual::new(problem2);
        solution1.update_objective("obj2", 100.0).unwrap();
        solution2.update_objective("obj2", 15.0).unwrap();
        solution1.update_constraint("c1", 0.5).unwrap();
        solution2.update_constraint("c1", 3.0).unwrap();
        assert_eq!(
            ParetoConstrainedDominance::compare(&solution1, &solution2).unwrap(),
            PreferredSolution::Second
        );
    }
}
