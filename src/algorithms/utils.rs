use crate::core::{Individual, Individuals, OError, VariableValue};
use crate::core::individual::IndividualsMut;
use crate::core::utils::{argsort, vector_max, vector_min};
use crate::operators::{BinaryComparisonOperator, ParetoConstrainedDominance, PreferredSolution};

/// Outputs of the non-dominated sort algorithm.
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

/// Non-dominated fast sorting for NSGA2 (with complexity $O(M * N^2)$, where `M` is the number of
/// objectives and `N` the number of individuals).
///
/// This sorts solutions into fronts and ranks the individuals based on the number of solutions
/// an individual dominates. Solutions that are not dominated by any other individuals will belong
/// to the first front. The method also stores the `rank` property into each individual; to retrieve
/// it, use `Individual::set_data("rank").unwrap()`.
///
/// Implemented based on paragraph 3A in:
/// > K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multi-objective genetic
/// > algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp.
/// > 182-197, April 2002, doi: 10.1109/4235.996017.
///
/// # Arguments
///
/// * `individuals`: The individuals to sort.
///
/// returns: `Result<NonDominatedSortResults, OError>`.
pub fn fast_non_dominated_sort(
    individuals: &mut [Individual],
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

/// Calculate the crowding distance (with complexity $O(M * log(N))$, where `M` is the number of
/// objectives and `N` the number of individuals). This set the distance on the individual's data,
/// to retrieve it, use `Individual::set_data("crowding_distance").unwrap()`.
/// > NOTE: the individuals must be a non-dominated front.
///
/// Implemented based on paragraph 3B in:
/// > K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multi-objective genetic
/// > algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp.
/// > 182-197, April 2002, doi: 10.1109/4235.996017.
///
/// # Arguments
///
/// * `individuals`: The individuals in a non-dominated front.
///
/// returns: `Result<(), OError>`
pub fn set_crowding_distance(mut individuals: &mut [Individual]) -> Result<(), OError> {
    let data_name = "crowding_distance";
    let inf = VariableValue::Real(f64::INFINITY);
    let total_individuals = individuals.len();

    // if there are enough point set distance to + infinite
    if total_individuals < 3 {
        for individual in individuals {
            individual.set_data(data_name, inf.clone());
        }
        return Ok(());
    }

    for individual in individuals.iter_mut() {
        individual.set_data(data_name, VariableValue::Real(0.0));
    }

    let problem = individuals.individual(0)?.problem();
    for obj_name in problem.objective_names() {
        let mut obj_values = individuals.objective_values(&obj_name)?;
        let delta_range = vector_max(&obj_values)? - vector_min(&obj_values)?;

        // set all to infinite if distance is too small
        if delta_range.abs() < f64::EPSILON {
            for individual in &mut *individuals {
                individual.set_data(data_name, inf.clone());
            }
        }

        // sort objectives and get indexes to map individuals to sorted objectives
        let sorted_idx = argsort(&obj_values);
        obj_values.sort_by(|a, b| a.total_cmp(b));

        // assign infinite distance to the boundary points
        individuals
            .individual_as_mut(sorted_idx[0])?
            .set_data(data_name, inf.clone());
        individuals
            .individual_as_mut(sorted_idx[total_individuals - 1])?
            .set_data(data_name, inf.clone());

        for obj_i in 1..(total_individuals - 1) {
            // get corresponding individual to sorted objective
            let ind_i = sorted_idx[obj_i];
            let current_distance = individuals
                .individual(ind_i)?
                .get_data(data_name)
                .unwrap_or(VariableValue::Real(0.0));

            if let VariableValue::Real(current_distance) = current_distance {
                let delta = (obj_values[obj_i + 1] - obj_values[obj_i - 1]) / delta_range;
                if delta.is_nan() {
                    return Err(OError::Generic(
                        format!("The calculated crowding distance increment was NaN likely due to wrong objective values. Numerator: {}, denominator: {}", 
                                obj_values[obj_i + 1] - obj_values[obj_i - 1], delta_range)
                    ));
                }
                individuals
                    .individual_as_mut(ind_i)?
                    .set_data(data_name, VariableValue::Real(current_distance + delta));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use float_cmp::assert_approx_eq;

    use crate::algorithms::utils::{fast_non_dominated_sort, set_crowding_distance};
    use crate::core::{
        BoundedNumber, Individual, Individuals, Objective, ObjectiveDirection, Problem,
        VariableType, VariableValue,
    };
    use crate::core::utils::dummy_evaluator;

    /// Create the individuals for a `N`-objective problem, where `N` is the number of items in
    /// the arrays of `objective_values`.
    ///
    /// # Arguments
    ///
    /// * `objective_values`: The objective values to set on the individuals. A number of
    /// individuals equal to the number of rows in this vector will be created.
    ///
    /// returns: `Vec<Individual>`
    fn create_individuals<const N: usize>(objective_values: Vec<[f64; N]>) -> Vec<Individual> {
        let mut objectives = Vec::new();
        for i in 0..N {
            objectives.push(Objective::new(
                format!("obj{i}").as_str(),
                ObjectiveDirection::Minimise,
            ));
        }
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X", 0.0, 2.0).unwrap(),
        )];
        let problem =
            Arc::new(Problem::new(objectives, variables, None, dummy_evaluator()).unwrap());

        // create the individuals
        let mut individuals: Vec<Individual> = Vec::new();
        for data in objective_values {
            let mut individual = Individual::new(problem.clone());
            for (i, obj_value) in data.into_iter().enumerate() {
                individual
                    .update_objective(format!("obj{i}").as_str(), obj_value)
                    .unwrap();
            }
            individuals.push(individual);
        }

        individuals
    }

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
        let mut individuals = create_individuals(objectives);
        let result = fast_non_dominated_sort(&mut individuals).unwrap();

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

        // calculate distance
        set_crowding_distance(&mut individuals).unwrap();
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
        let mut individuals = create_individuals(objectives);
        let result = fast_non_dominated_sort(&mut individuals).unwrap();

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

    #[test]
    /// Test the crowding distance algorithm (not enough points).
    fn test_crowding_distance_empty() {
        let objectives = vec![[0.0, 0.0], [50.0, 50.0]];
        let mut individuals = create_individuals(objectives);
        set_crowding_distance(&mut individuals).unwrap();
        for i in individuals {
            assert_eq!(
                i.get_data("crowding_distance").unwrap(),
                VariableValue::Real(f64::INFINITY)
            );
        }
    }

    #[test]
    /// Test the crowding distance algorithm (3 points).
    fn test_crowding_distance_3_points() {
        // 3 points
        let scenarios = vec![
            vec![[0.0, 0.0], [-100.0, 100.0], [200.0, -200.0]],
            vec![[25.0, 25.0], [-100.0, 100.0], [200.0, -200.0]],
        ];
        for objectives in scenarios {
            let mut individuals = create_individuals(objectives);
            set_crowding_distance(&mut individuals).unwrap();

            assert_eq!(
                individuals
                    .as_mut_slice()
                    .individual(0)
                    .unwrap()
                    .get_data("crowding_distance")
                    .unwrap(),
                VariableValue::Real(2.0)
            );
            // boundaries
            assert_eq!(
                individuals
                    .as_mut_slice()
                    .individual(1)
                    .unwrap()
                    .get_data("crowding_distance")
                    .unwrap(),
                VariableValue::Real(f64::INFINITY)
            );
            assert_eq!(
                individuals
                    .as_mut_slice()
                    .individual(2)
                    .unwrap()
                    .get_data("crowding_distance")
                    .unwrap(),
                VariableValue::Real(f64::INFINITY)
            );
        }
    }

    #[test]
    /// Test the crowding distance algorithm (3 objectives).
    fn test_crowding_distance_3_obj() {
        let objectives = vec![[0.0, 0.0, 0.0], [-1.0, 1.0, 2.0], [2.0, -2.0, -2.0]];
        let mut individuals = create_individuals(objectives);
        set_crowding_distance(&mut individuals).unwrap();

        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(0)
                .unwrap()
                .get_data("crowding_distance")
                .unwrap(),
            VariableValue::Real(3.0)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(1)
                .unwrap()
                .get_data("crowding_distance")
                .unwrap(),
            VariableValue::Real(f64::INFINITY)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(2)
                .unwrap()
                .get_data("crowding_distance")
                .unwrap(),
            VariableValue::Real(f64::INFINITY)
        );
    }

    #[test]
    /// Test the crowding distance algorithm (4 points).
    fn test_crowding_distance_4points() {
        let objectives = vec![
            [0.0, 0.0],
            [100.0, -100.0],
            [200.0, -200.0],
            [400.0, -400.0],
        ];
        let mut individuals = create_individuals(objectives);
        set_crowding_distance(&mut individuals).unwrap();

        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(0)
                .unwrap()
                .get_data("crowding_distance")
                .unwrap(),
            VariableValue::Real(f64::INFINITY)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(1)
                .unwrap()
                .get_data("crowding_distance")
                .unwrap(),
            VariableValue::Real(1.0)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(2)
                .unwrap()
                .get_data("crowding_distance")
                .unwrap(),
            VariableValue::Real(1.5)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(3)
                .unwrap()
                .get_data("crowding_distance")
                .unwrap(),
            VariableValue::Real(f64::INFINITY)
        );
    }

    #[test]
    /// Test the crowding distance algorithm (6 points).
    fn test_crowding_distance_6points() {
        let objectives = vec![
            [1.1, 8.1],
            [2.1, 6.1],
            [3.1, 4.1],
            [5.1, 3.1],
            [8.1, 2.1],
            [11.1, 1.1],
        ];
        let mut individuals = create_individuals(objectives);
        set_crowding_distance(&mut individuals).unwrap();

        let expected = [
            f64::INFINITY,
            0.7714285714285714,
            0.728571429,
            0.785714286,
            0.885714286,
            f64::INFINITY,
        ];
        for (idx, value) in expected.into_iter().enumerate() {
            assert_approx_eq!(
                f64,
                individuals
                    .as_mut_slice()
                    .individual(idx)
                    .unwrap()
                    .get_data("crowding_distance")
                    .unwrap()
                    .as_real()
                    .unwrap(),
                VariableValue::Real(value).as_real().unwrap(),
                epsilon = 0.001
            );
        }
    }
}
