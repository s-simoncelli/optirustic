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

#[cfg(test)]
mod test_problems {
    use float_cmp::approx_eq;

    use optirustic_macros::test_with_retries;

    use crate::algorithms::{
        AdaptiveNSGA3, Algorithm, MaxGenerationValue, NSGA3Arg, Nsga3NumberOfIndividuals,
        StoppingConditionType,
    };
    use crate::core::builtin_problems::DTLZ1Problem;
    use crate::core::test_utils::assert_approx_array_eq;
    use crate::operators::{PolynomialMutationArgs, SimulatedBinaryCrossoverArgs};
    use crate::utils::NumberOfPartitions;

    #[test_with_retries(10)]
    /// Test the inverted DTLZ1 problem with M=3 and MaxGeneration = 400
    fn test_inverted_dtlz1_obj_3() {
        let k: usize = 5;
        let number_variables: usize = 3 + k - 1;
        let problem = DTLZ1Problem::create(number_variables, 3, true).unwrap();
        let number_of_partitions = NumberOfPartitions::OneLayer(12);
        let pop_size: usize = 92;
        let expected_ref_points: usize = 91;

        let crossover_operator_options = SimulatedBinaryCrossoverArgs {
            distribution_index: 30.0,
            ..SimulatedBinaryCrossoverArgs::default()
        };
        let mutation_operator_options = PolynomialMutationArgs::default(&problem);

        let args = NSGA3Arg {
            number_of_individuals: Nsga3NumberOfIndividuals::Custom(pop_size),
            number_of_partitions,
            crossover_operator_options: Some(crossover_operator_options),
            mutation_operator_options: Some(mutation_operator_options),
            stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(400)),
            parallel: None,
            export_history: None,
            seed: Some(1),
        };

        let mut algo = AdaptiveNSGA3::new(problem, args).unwrap();
        assert_eq!(algo.reference_points().len(), expected_ref_points);

        algo.run().unwrap();
        let results = algo.get_results();

        let expected_vars = vec![0.5; number_variables];
        let mut invalid_individuals: usize = 0;
        for ind in &results.individuals {
            let obj_sum: f64 = ind.get_objective_values().unwrap().iter().sum();
            // objective target sum is 1
            let outside_range_data = approx_eq!(f64, obj_sum, 1.0, epsilon = 0.01);
            if !outside_range_data {
                invalid_individuals += 1;
            }

            // All variables in x_M must be 0.5
            let vars: Vec<f64> = ((number_variables - k + 1)..=number_variables)
                .map(|i| {
                    ind.get_variable_value(format!("x{i}").as_str())
                        .unwrap()
                        .as_real()
                        .unwrap()
                })
                .collect();
            assert_approx_array_eq(&vars, &expected_vars, Some(0.01));
        }

        // about 90% of solutions are ideal
        if invalid_individuals > 10 {
            panic!("Found {invalid_individuals} individuals not meeting the ideal solution");
        }

        // all new associated reference points have at least one associated individuals. These
        // are only the points within the optimal Pareto front (when objective is <= 0.5)
        let new_assoc_ref_points: Vec<_> = results
            .individuals
            .iter()
            .filter_map(|i| {
                let index = i
                    .get_data("reference_point_index")
                    .unwrap()
                    .as_usize()
                    .unwrap();
                if index > 92 {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();

        for (ri, r) in results.additional_data["reference_points"]
            .as_data_vec()
            .unwrap()
            .iter()
            .enumerate()
        {
            if r.as_f64_vec().unwrap().iter().all(|coord| *coord <= 0.5) && ri > 92 {
                if !new_assoc_ref_points.contains(&ri) {
                    panic!(
                        "Reference point #{ri} ({:?}) is not associated with any individual",
                        ri
                    );
                }
            }
        }
    }
}
