use crate::core::{Individual, OError, VariableValue};
use crate::operators::{BinaryComparisonOperator, ParetoConstrainedDominance, PreferredSolution};

/// Outputs of the non-dominated sort algorithm.
#[derive(Debug)]
pub struct NonDominatedSortResults {
    /// A vector containing sub-vectors. Each child vector represents a front (with the first being
    /// the primary non-dominated front with solutions of rank 1); each child vector contains
    /// the individuals belonging to that front.
    pub fronts: Vec<Vec<Individual>>,
    /// This is [`NonDominatedSortResults::fronts`], but the individuals are given as indexes
    /// instead of references. Each index refers to the vector of individuals passed to
    /// [`fast_non_dominated_sort`].
    pub front_indexes: Vec<Vec<usize>>,
    /// Number of individuals that dominates a solution at a given vector index. When the counter
    /// is 0, the solution is non-dominated. This is `n_p` in the paper.
    pub domination_counter: Vec<usize>,
}

/// Non-dominated fast sorting from NSGA2 paper (with complexity $O(M * N^2)$, where `M` is the
/// number of objectives and `N` the number of individuals).
///
/// This sorts solutions into fronts and ranks the individuals based on the number of solutions
/// an individual dominates. Solutions that are not dominated by any other individuals will belong
/// to the first front. The method also stores the `rank` property into each individual; to retrieve
/// it, use `Individual::get_data("rank").unwrap()`.
///
/// Implemented based on paragraph 3A in:
/// > K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multi-objective genetic
/// > algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp.
/// > 182-197, April 2002, doi: 10.1109/4235.996017.
///
/// # Arguments
///
/// * `individuals`: The individuals to sort by dominance.
/// * `first_front_only`: Return the first front only with the rank 1 (i.e. containing only
/// non-dominated individuals). If you need only the first front set this to true to avoid
/// ranking the remaining individuals.
///
/// returns: `Result<NonDominatedSortResults, OError>`.
pub fn fast_non_dominated_sort(
    individuals: &mut [Individual],
    first_front_only: bool,
) -> Result<NonDominatedSortResults, OError> {
    if individuals.len() < 2 {
        return Err(OError::SurvivalOperator(
            "fast non-dominated sort".to_string(),
            format!(
                "At least 2 individuals are needed for sorting, but {} given",
                individuals.len()
            ),
        ));
    }

    // this set contains all the individuals being dominated by an individual `p`.This is `S_p` in
    // the paper
    let mut dominated_solutions: Vec<Vec<usize>> = individuals.iter().map(|_| Vec::new()).collect();
    // number of individuals that dominates `p`. When the counter is 0, `p` is non-dominated. This
    // is `n_p` in the paper
    let mut domination_counter: Vec<usize> = individuals.iter().map(|_| 0).collect();

    // the front of given rank containing non-dominated solutions
    let mut current_front: Vec<usize> = Vec::new();
    // the vector with all fronts of sorted ranks. The first item has rank 1 and subsequent elements
    // have increasing rank
    let mut all_fronts: Vec<Vec<usize>> = Vec::new();

    for pi in 0..individuals.len() {
        for qi in pi..individuals.len() {
            match ParetoConstrainedDominance::compare(&individuals[pi], &individuals[qi])? {
                PreferredSolution::First => {
                    // `p` dominates `q` - add `q` to the set of solutions dominated by `p`
                    dominated_solutions[pi].push(qi);
                    domination_counter[qi] += 1;
                }
                PreferredSolution::Second => {
                    // q dominates p
                    dominated_solutions[qi].push(pi);
                    domination_counter[pi] += 1;
                }
                PreferredSolution::MutuallyPreferred => {
                    // skip this
                }
            }
        }

        // the solution `p` is non-dominated by any other and this solution belongs to the first
        // front whose items have rank 1
        if domination_counter[pi] == 0 {
            current_front.push(pi);
            individuals[pi].set_data("rank", VariableValue::Integer(1));
        }
    }

    // early return
    if first_front_only {
        let first_front = current_front
            .iter()
            .map(|idx| individuals[*idx].clone())
            .collect();
        return Ok(NonDominatedSortResults {
            fronts: vec![first_front],
            front_indexes: vec![current_front],
            domination_counter,
        });
    }
    all_fronts.push(current_front.clone());
    let e_domination_counter = domination_counter.clone();

    // collect the other fronts
    let mut i = 1;
    loop {
        let mut next_front: Vec<usize> = Vec::new();
        // loop individuals in the current non-dominated front
        for pi in current_front.iter() {
            // loop solutions that are dominated by `p` in the current front
            for qi in dominated_solutions[*pi].iter() {
                // decrement the domination count for individual `q`
                domination_counter[*qi] -= 1;

                // if counter is 0 then none of the individuals in the subsequent fronts are
                // dominated by `p` and `q` belongs to the next front
                if domination_counter[*qi] == 0 {
                    next_front.push(*qi);
                    individuals[*qi].set_data("rank", VariableValue::Integer(i + 1));
                }
            }
        }
        i += 1;

        // stop when all solutions have been ranked
        if next_front.is_empty() {
            break;
        }

        all_fronts.push(next_front.clone());
        current_front = next_front;
    }

    // map index to individuals
    let mut fronts: Vec<Vec<Individual>> = Vec::new();
    for front in &all_fronts {
        let mut sub_front: Vec<Individual> = Vec::new();
        for i in front {
            sub_front.push(individuals[*i].clone());
        }
        fronts.push(sub_front);
    }

    Ok(NonDominatedSortResults {
        fronts,
        front_indexes: all_fronts,
        domination_counter: e_domination_counter,
    })
}

#[cfg(test)]
mod test {
    use crate::core::{ObjectiveDirection, VariableValue};
    use crate::core::utils::individuals_from_obj_values_dummy;
    use crate::utils::fast_non_dominated_sort;

    #[test]
    /// Test the non-dominated sorting. The resulting fronts and ranks were manually calculated by
    /// plotting the objective values.
    fn test_sorting_2obj() {
        let objectives = vec![
            [1.1, 8.1],
            [2.1, 6.1],
            [3.1, 4.1],
            [3.1, 7.1],
            [5.1, 3.1],
            [5.1, 5.1],
            [7.1, 7.1],
            [8.1, 2.1],
            [10.1, 6.1],
            [11.1, 1.1],
            [11.1, 3.1],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
        );
        let result = fast_non_dominated_sort(&mut individuals, false).unwrap();

        // non-dominated front
        let expected_first = vec![0, 1, 2, 4, 7, 9];
        assert_eq!(result.front_indexes[0], expected_first);

        // check rank
        for idx in &expected_first {
            assert_eq!(
                individuals[*idx].get_data("rank").unwrap(),
                VariableValue::Integer(1)
            );
        }

        // other fronts
        let expected_second = vec![3, 5, 10];
        assert_eq!(result.front_indexes[1], expected_second);
        for idx in expected_second {
            assert_eq!(
                individuals[idx].get_data("rank").unwrap(),
                VariableValue::Integer(2)
            );
        }

        let expected_third = vec![6, 8];
        assert_eq!(result.front_indexes[2], expected_third);
        for idx in expected_third {
            assert_eq!(
                individuals[idx].get_data("rank").unwrap(),
                VariableValue::Integer(3)
            );
        }

        // check counter for some solutions
        for idx in expected_first {
            assert_eq!(result.domination_counter[idx], 0);
        }
        // by 6 and 8
        assert_eq!(result.domination_counter[5], 2);
        // by 1, 2, 4, 5 and 7
        assert_eq!(result.domination_counter[8], 5);
        // by 0 and 1
        assert_eq!(result.domination_counter[3], 2);
    }

    #[test]
    /// Test the non-dominated sorting when objective #1 is maximised.
    fn test_sorting_2obj_max_obj1() {
        let objectives = vec![
            [11.1, 8.1],
            [8.1, 6.1],
            [5.1, 4.1],
            [3.1, 3.1],
            [2.1, 2.1],
            [1.1, 1.1],
            [0.0, 5.1],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[ObjectiveDirection::Maximise, ObjectiveDirection::Minimise],
        );
        let result = fast_non_dominated_sort(&mut individuals, false).unwrap();

        // non-dominated front
        let expected_first = (0..=5).collect::<Vec<usize>>();
        assert_eq!(result.front_indexes[0], expected_first);

        // check rank
        for idx in &expected_first {
            assert_eq!(
                individuals[*idx].get_data("rank").unwrap(),
                VariableValue::Integer(1)
            );
        }

        // other fronts
        let expected_second = vec![6];
        assert_eq!(result.front_indexes[1], expected_second);
    }

    #[test]
    /// Test the non-dominated sorting when objective #2 is maximised.
    fn test_sorting_2obj_max_obj2() {
        let objectives = vec![
            [11.1, 8.1],
            [8.1, 6.1],
            [5.1, 4.1],
            [3.1, 3.1],
            [2.1, 2.1],
            [1.1, 1.1],
            [0.0, 5.1],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Maximise],
        );
        let result = fast_non_dominated_sort(&mut individuals, false).unwrap();

        // non-dominated front
        let expected_first = vec![0, 1, 6];
        assert_eq!(result.front_indexes[0], expected_first);

        // check rank
        for idx in &expected_first {
            assert_eq!(
                individuals[*idx].get_data("rank").unwrap(),
                VariableValue::Integer(1)
            );
        }

        // other fronts
        let expected_second = vec![2, 3, 4, 5];
        assert_eq!(result.front_indexes[1], expected_second);
    }

    #[test]
    /// Test the non-dominated sorting. The resulting fronts and ranks were manually calculated by
    /// plotting the objective values.
    fn test_sorting_3obj() {
        let objectives = vec![
            [2.1, 3.1, 4.1],
            [-1.1, 4.1, 8.1],
            [0.1, -1.1, -2.1],
            [0.1, 0.1, 0.1],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[
                ObjectiveDirection::Minimise,
                ObjectiveDirection::Minimise,
                ObjectiveDirection::Minimise,
            ],
        );
        let result = fast_non_dominated_sort(&mut individuals, false).unwrap();

        // non-dominated front
        let expected_first = vec![1, 2];
        assert_eq!(result.front_indexes[0], expected_first);

        // check rank
        for idx in &expected_first {
            assert_eq!(
                individuals[*idx].get_data("rank").unwrap(),
                VariableValue::Integer(1)
            );
        }

        // other fronts
        let expected_second = vec![3];
        assert_eq!(result.front_indexes[1], expected_second);
        for idx in expected_second {
            assert_eq!(
                individuals[idx].get_data("rank").unwrap(),
                VariableValue::Integer(2)
            );
        }

        let expected_third = vec![0];
        assert_eq!(result.front_indexes[2], expected_third);
        for idx in expected_third {
            assert_eq!(
                individuals[idx].get_data("rank").unwrap(),
                VariableValue::Integer(3)
            );
        }

        // check counter for some solutions
        for idx in expected_first {
            assert_eq!(result.domination_counter[idx], 0);
        }
        assert_eq!(result.domination_counter[0], 2);
        assert_eq!(result.domination_counter[3], 1);
    }
}
