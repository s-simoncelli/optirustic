use std::fmt::{Display, Formatter};
use std::ops::Rem;
use std::path::PathBuf;

use log::{debug, info};
use rand::RngCore;

use optirustic_macros::{as_algorithm, as_algorithm_args, impl_algorithm_trait_items};

use crate::algorithms::Algorithm;
use crate::core::utils::get_rng;
use crate::core::{DataValue, Individual, Individuals, IndividualsMut, OError};
use crate::operators::{
    Crossover, CrowdedComparison, Mutation, PolynomialMutation, PolynomialMutationArgs, Selector,
    SimulatedBinaryCrossover, SimulatedBinaryCrossoverArgs, TournamentSelector,
};
use crate::utils::{argsort, fast_non_dominated_sort, vector_max, vector_min, Sort};

/// The data key where the crowding distance is stored for each [`Individual`].
const CROWDING_DIST_KEY: &str = "crowding_distance";

/// Input arguments for the NSGA2 algorithm.
#[as_algorithm_args]
pub struct NSGA2Arg {
    /// The number of individuals to use in the population. This must be a multiple of `2`.
    pub number_of_individuals: usize,
    /// The options of the Simulated Binary Crossover (SBX) operator. This operator is used to
    /// generate new children by recombining the variables of parent solutions. This defaults to
    /// [`SimulatedBinaryCrossoverArgs::default()`].
    pub crossover_operator_options: Option<SimulatedBinaryCrossoverArgs>,
    /// The options to Polynomial Mutation (PM) operator used to mutate the variables of an
    /// individual. This defaults to [`PolynomialMutationArgs::default()`],
    /// with a distribution index or index parameter of `20` and variable probability equal `1`
    /// divided by the number of real variables in the problem (i.e., each variable will have the
    /// same probability of being mutated).
    pub mutation_operator_options: Option<PolynomialMutationArgs>,
    /// Instead of initialising the population with random variables, see the initial population
    /// with  the variable values from a JSON files exported with this tool. This option lets you
    /// restart the evolution from a previous generation; you can use any history file (exported
    /// when the field `export_history`) or the file exported when the stopping condition was reached.
    pub resume_from_file: Option<PathBuf>,
}

/// The Non-dominated Sorting Genetic Algorithm (NSGA2).
///
/// Implemented based on:
/// > K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multi-objective genetic
/// > algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp.
/// > 182-197, April 2002, doi: 10.1109/4235.996017.
///
/// See: <https://doi.org/10.1109/4235.996017>.
///
/// # Examples
/// ## Solve the Schaffer’s problem
/// ```rust
#[doc = include_str!("../../examples/nsga2_sch.rs")]
/// ```
/// ## Solve the ZDT1 problem
/// ```rust
#[doc = include_str!("../../examples/nsga2_zdt1.rs")]
/// ```
#[as_algorithm(NSGA2Arg)]
pub struct NSGA2 {
    /// The operator to use to select the individuals for reproduction. This is a binary tournament
    /// selector ([`TournamentSelector`]) with the [`CrowdedComparison`] comparison operator.
    selector_operator: TournamentSelector<CrowdedComparison>,
    /// The SBX operator to use to generate a new children by recombining the variables of parent
    /// solutions.
    crossover_operator: SimulatedBinaryCrossover,
    /// The PM operator to use to mutate the variables of an individual.
    mutation_operator: PolynomialMutation,
    /// The seed to use.
    rng: Box<dyn RngCore>,
}

impl NSGA2 {
    /// Initialise the NSGA2 algorithm.
    ///
    /// # Arguments
    ///
    /// * `problem`: The problem being solved.
    /// * `args`: The [`NSGA2Arg`] arguments to customise the algorithm behaviour.
    ///
    /// returns: `NSGA2`.
    pub fn new(problem: Problem, options: NSGA2Arg) -> Result<Self, OError> {
        let name = "NSGA2".to_string();
        if options.number_of_individuals < 3 {
            return Err(OError::AlgorithmInit(
                name,
                "The population size must have at least 3 individuals".to_string(),
            ));
        }
        // force the population size as multiple of 2 so that the new number of generated offsprings
        // matches `number_of_individuals`
        if options.number_of_individuals.rem(2) != 0 {
            return Err(OError::AlgorithmInit(
                name,
                "The population size must be a multiple of 2".to_string(),
            ));
        }

        let nsga2_args = options.clone();
        let problem = Arc::new(problem);
        let population = if let Some(init_file) = options.resume_from_file {
            info!("Loading initial population from {:?}", init_file);
            NSGA2::seed_population_from_file(
                problem.clone(),
                &name,
                options.number_of_individuals,
                &init_file,
            )?
        } else {
            info!("Created initial random population");
            Population::init(problem.clone(), options.number_of_individuals)
        };

        let mutation_options = match options.mutation_operator_options {
            Some(o) => o,
            None => PolynomialMutationArgs::default(problem.clone().as_ref()),
        };
        let mutation_operator = PolynomialMutation::new(mutation_options.clone())?;

        let crossover_options = options.crossover_operator_options.unwrap_or_default();
        let crossover_operator = SimulatedBinaryCrossover::new(crossover_options.clone())?;

        info!(
            "{}",
            Self::algorithm_option_str(&problem, &crossover_options, &mutation_options)
        );

        Ok(Self {
            number_of_individuals: options.number_of_individuals,
            problem,
            population,
            selector_operator: TournamentSelector::<CrowdedComparison>::new(2),
            crossover_operator,
            mutation_operator,
            generation: 0,
            nfe: 0,
            stopping_condition: options.stopping_condition,
            start_time: Instant::now(),
            parallel: options.parallel.unwrap_or(true),
            export_history: options.export_history,
            rng: get_rng(options.seed),
            args: nsga2_args,
        })
    }

    /// Get a string listing the algorithm options.
    ///
    /// # Arguments
    ///
    /// * `problem`: The problem.
    /// * `crossover_options`: The crossover operator options.
    /// * `mutation_options`: The mutation operator options.
    ///
    /// returns: `String`
    pub fn algorithm_option_str(
        problem: &Arc<Problem>,
        crossover_options: &SimulatedBinaryCrossoverArgs,
        mutation_options: &PolynomialMutationArgs,
    ) -> String {
        let mut log_opts: String = "Algorithm options are:\n".to_owned();
        log_opts.push_str(
            format!("\t* Number of variables {:>13}\n\t* Number of objectives {:>12}\n\t* Number of constraints {:>11}\n",
                    problem.number_of_variables(),
                    problem.number_of_objectives(),
                    problem.number_of_constraints()
            ).as_str()
        );
        log_opts.push_str(
            format!(
                "\t* Crossover distribution index {:>5}\n\t* Crossover probability {:>11}\n\t* Crossover var probability {:>9}\n",
                crossover_options.distribution_index, crossover_options.crossover_probability, crossover_options.variable_probability,
            )
                .as_str(),
        );
        log_opts.push_str(
            format!(
                "\t* Mutation index parameter {:>9}\n\t* Mutation var probability {:>10}",
                mutation_options.index_parameter, crossover_options.variable_probability,
            )
            .as_str(),
        );
        log_opts
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
    fn set_crowding_distance(mut individuals: &mut [Individual]) -> Result<(), OError> {
        let inf = DataValue::Real(f64::MAX); // do not use INF because is not supported by serde
        let total_individuals = individuals.len();

        // if there are enough point set distance to + infinite
        if total_individuals < 3 {
            for individual in individuals {
                individual.set_data(CROWDING_DIST_KEY, inf.clone());
            }
            debug!("Setting crowding distance to Inf for all individuals. At least 3 individuals are needed");

            return Ok(());
        }

        for individual in individuals.iter_mut() {
            individual.set_data(CROWDING_DIST_KEY, DataValue::Real(0.0));
        }

        let problem = individuals.individual(0)?.problem();
        for obj_name in problem.objective_names() {
            let mut obj_values = individuals.objective_values(&obj_name)?;
            let delta_range = vector_max(&obj_values)? - vector_min(&obj_values)?;

            // set all to infinite if distance is too small
            if delta_range.abs() < f64::EPSILON {
                for individual in &mut *individuals {
                    individual.set_data(CROWDING_DIST_KEY, inf.clone());
                }
                debug!("Setting crowding distance to Inf for all individuals. The min/max range is too small");
                return Ok(());
            }

            // sort objectives and get indexes to map individuals to sorted objectives
            let sorted_idx = argsort(&obj_values, Sort::Ascending);
            obj_values.sort_by(|a, b| a.total_cmp(b));

            // assign infinite distance to the boundary points
            individuals
                .individual_as_mut(sorted_idx[0])?
                .set_data(CROWDING_DIST_KEY, inf.clone());
            individuals
                .individual_as_mut(sorted_idx[total_individuals - 1])?
                .set_data(CROWDING_DIST_KEY, inf.clone());

            for obj_i in 1..(total_individuals - 1) {
                // get the corresponding individual to sorted objective
                let ind_i = sorted_idx[obj_i];
                let current_distance = individuals
                    .individual(ind_i)?
                    .get_data(CROWDING_DIST_KEY)
                    .unwrap_or(DataValue::Real(0.0));

                if let DataValue::Real(current_distance) = current_distance {
                    let delta = (obj_values[obj_i + 1] - obj_values[obj_i - 1]) / delta_range;
                    individuals
                        .individual_as_mut(ind_i)?
                        .set_data(CROWDING_DIST_KEY, DataValue::Real(current_distance + delta));
                }
            }
        }

        Ok(())
    }
}

/// Implementation of Section IIIC of the paper.
#[impl_algorithm_trait_items(NSGA2Arg)]
impl Algorithm<NSGA2Arg> for NSGA2 {
    /// This assesses the initial random population and sets the individual's ranks and crowding
    /// distance needed in [`self.evolve`].
    ///
    /// return: `Result<(), OError>`
    fn initialise(&mut self) -> Result<(), OError> {
        info!("Evaluating initial population");
        if self.parallel {
            NSGA2::do_parallel_evaluation(self.population.individuals_as_mut(), &mut self.nfe)?;
        } else {
            NSGA2::do_evaluation(self.population.individuals_as_mut(), &mut self.nfe)?;
        }

        debug!("Calculating rank");
        fast_non_dominated_sort(self.population.individuals_as_mut(), false)?;

        debug!("Calculating crowding distance");
        NSGA2::set_crowding_distance(self.population.individuals_as_mut())?;

        info!("Initial evaluation completed");
        self.generation += 1;

        Ok(())
    }

    fn evolve(&mut self) -> Result<(), OError> {
        // Create the new population, based on the population at the previous time-step, of size
        // self.number_of_individuals. The loop adds two individuals at the time.
        debug!("Generating new population (selection + crossover + mutation)");
        let mut offsprings: Vec<Individual> = Vec::new();
        for _ in 0..self.number_of_individuals / 2 {
            let parents =
                self.selector_operator
                    .select(self.population.individuals(), 2, &mut self.rng)?;

            // generate the 2 children with crossover
            let children = self.crossover_operator.generate_offsprings(
                &parents[0],
                &parents[1],
                &mut self.rng,
            )?;

            // mutate them
            offsprings.push(
                self.mutation_operator
                    .mutate_offspring(&children.child1, &mut self.rng)?,
            );
            offsprings.push(
                self.mutation_operator
                    .mutate_offspring(&children.child2, &mut self.rng)?,
            );
        }
        debug!("Combining parents and offsprings in new population");
        self.population.add_new_individuals(offsprings);
        debug!("New population size is {}", self.population.len());

        debug!("Evaluating population");
        if self.parallel {
            NSGA2::do_parallel_evaluation(self.population.individuals_as_mut(), &mut self.nfe)?;
        } else {
            NSGA2::do_evaluation(self.population.individuals_as_mut(), &mut self.nfe)?;
        }
        debug!("Evaluation done");

        debug!("Calculating fronts and ranks for new population");
        let sorting_results = fast_non_dominated_sort(self.population.individuals_as_mut(), false)?;
        debug!("Collected {} fronts", sorting_results.fronts.len());

        debug!("Selecting best individuals");
        let mut new_population = Population::new();

        // This selects the best individuals that will form the new population which contains the
        // population at the previous generation and the new offsprings. The new population is created
        // by keeping adding ranked non-dominated fronts until the population size almost reaches
        // `self.number_of_individuals`. When the last front does not fit, the individuals are then
        // added based on their crowding distance.
        //
        // This implements the algorithm at the bottom of page 186 in Deb et al. (2002).
        let mut last_front: Option<Vec<Individual>> = None;
        for (fi, front) in sorting_results.fronts.into_iter().enumerate() {
            if new_population.len() + front.len() <= self.number_of_individuals {
                debug!("Adding front #{} (size: {})", fi + 1, front.len());
                new_population.add_new_individuals(front);
            } else if new_population.len() == self.number_of_individuals {
                debug!("Population reached target size");
                break;
            } else {
                debug!(
                    "Population almost full ({} individuals)",
                    new_population.len()
                );
                last_front = Some(front.clone());
                break;
            }
        }

        // Complete the population with the last front
        if let Some(mut last_front) = last_front {
            NSGA2::set_crowding_distance(&mut last_front)?;

            // Sort in descending order. Prioritise individuals with the largest distance to
            // prevent crowding
            last_front.sort_by(|i, o| {
                i.get_data(CROWDING_DIST_KEY)
                    .unwrap()
                    .as_real()
                    .unwrap()
                    .total_cmp(&o.get_data(CROWDING_DIST_KEY).unwrap().as_real().unwrap())
            });
            last_front.reverse();

            // add the items to complete the population
            last_front.truncate(self.number_of_individuals - new_population.len());
            new_population.add_new_individuals(last_front);
        }

        // update the population and the distance for the CrowdedComparison operator at the next
        // loop
        self.population = new_population;
        NSGA2::set_crowding_distance(self.population.individuals_as_mut())?;

        self.generation += 1;
        Ok(())
    }
}

#[cfg(test)]
mod test_sorting {
    use float_cmp::assert_approx_eq;

    use crate::algorithms::nsga2::CROWDING_DIST_KEY;
    use crate::algorithms::NSGA2;
    use crate::core::test_utils::individuals_from_obj_values_dummy;
    use crate::core::{DataValue, Individuals, ObjectiveDirection};

    #[test]
    /// Test the crowding distance algorithm (not enough points).
    fn test_crowding_distance_not_enough_points() {
        let objectives = vec![vec![0.0, 0.0], vec![50.0, 50.0]];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
            None,
        );
        NSGA2::set_crowding_distance(&mut individuals).unwrap();
        for i in individuals {
            assert_eq!(
                i.get_data(CROWDING_DIST_KEY).unwrap(),
                DataValue::Real(f64::MAX)
            );
        }
    }

    #[test]
    /// Test the crowding distance algorithm (min and max of objective is equal).
    fn test_crowding_distance_min_max_range() {
        let objectives = vec![
            vec![10.0, 20.0],
            vec![10.0, 20.0],
            vec![10.0, 20.0],
            vec![10.0, 20.0],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
            None,
        );
        NSGA2::set_crowding_distance(&mut individuals).unwrap();
        for i in individuals {
            assert_eq!(
                i.get_data(CROWDING_DIST_KEY).unwrap(),
                DataValue::Real(f64::MAX)
            );
        }
    }

    #[test]
    /// Test the crowding distance algorithm (3 points).
    fn test_crowding_distance_3_points() {
        // 3 points
        let scenarios = vec![
            vec![vec![0.0, 0.0], vec![-100.0, 100.0], vec![200.0, -200.0]],
            vec![vec![25.0, 25.0], vec![-100.0, 100.0], vec![200.0, -200.0]],
        ];
        for objectives in scenarios {
            let mut individuals = individuals_from_obj_values_dummy(
                &objectives,
                &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
                None,
            );
            NSGA2::set_crowding_distance(&mut individuals).unwrap();

            assert_eq!(
                individuals
                    .as_mut_slice()
                    .individual(0)
                    .unwrap()
                    .get_data(CROWDING_DIST_KEY)
                    .unwrap(),
                DataValue::Real(2.0)
            );
            // boundaries
            assert_eq!(
                individuals
                    .as_mut_slice()
                    .individual(1)
                    .unwrap()
                    .get_data(CROWDING_DIST_KEY)
                    .unwrap(),
                DataValue::Real(f64::MAX)
            );
            assert_eq!(
                individuals
                    .as_mut_slice()
                    .individual(2)
                    .unwrap()
                    .get_data(CROWDING_DIST_KEY)
                    .unwrap(),
                DataValue::Real(f64::MAX)
            );
        }
    }

    #[test]
    /// Test the crowding distance algorithm (3 objectives).
    fn test_crowding_distance_3_obj() {
        let objectives = vec![
            vec![0.0, 0.0, 0.0],
            vec![-1.0, 1.0, 2.0],
            vec![2.0, -2.0, -2.0],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[
                ObjectiveDirection::Minimise,
                ObjectiveDirection::Minimise,
                ObjectiveDirection::Minimise,
            ],
            None,
        );
        NSGA2::set_crowding_distance(&mut individuals).unwrap();

        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(0)
                .unwrap()
                .get_data(CROWDING_DIST_KEY)
                .unwrap(),
            DataValue::Real(3.0)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(1)
                .unwrap()
                .get_data(CROWDING_DIST_KEY)
                .unwrap(),
            DataValue::Real(f64::MAX)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(2)
                .unwrap()
                .get_data(CROWDING_DIST_KEY)
                .unwrap(),
            DataValue::Real(f64::MAX)
        );
    }

    #[test]
    /// Test the crowding distance algorithm (4 points).
    fn test_crowding_distance_4points() {
        let objectives = vec![
            vec![0.0, 0.0],
            vec![100.0, -100.0],
            vec![200.0, -200.0],
            vec![400.0, -400.0],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
            None,
        );
        NSGA2::set_crowding_distance(&mut individuals).unwrap();

        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(0)
                .unwrap()
                .get_data(CROWDING_DIST_KEY)
                .unwrap(),
            DataValue::Real(f64::MAX)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(1)
                .unwrap()
                .get_data(CROWDING_DIST_KEY)
                .unwrap(),
            DataValue::Real(1.0)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(2)
                .unwrap()
                .get_data(CROWDING_DIST_KEY)
                .unwrap(),
            DataValue::Real(1.5)
        );
        assert_eq!(
            individuals
                .as_mut_slice()
                .individual(3)
                .unwrap()
                .get_data(CROWDING_DIST_KEY)
                .unwrap(),
            DataValue::Real(f64::MAX)
        );
    }

    #[test]
    /// Test the crowding distance algorithm (6 points).
    fn test_crowding_distance_6points() {
        let objectives = vec![
            vec![1.1, 8.1],
            vec![2.1, 6.1],
            vec![3.1, 4.1],
            vec![5.1, 3.1],
            vec![8.1, 2.1],
            vec![11.1, 1.1],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &objectives,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
            None,
        );
        NSGA2::set_crowding_distance(&mut individuals).unwrap();

        let expected = [
            f64::MAX,
            0.7714285714285714,
            0.728571429,
            0.785714286,
            0.885714286,
            f64::MAX,
        ];
        for (idx, value) in expected.into_iter().enumerate() {
            assert_approx_eq!(
                f64,
                individuals
                    .as_mut_slice()
                    .individual(idx)
                    .unwrap()
                    .get_data(CROWDING_DIST_KEY)
                    .unwrap()
                    .as_real()
                    .unwrap(),
                DataValue::Real(value).as_real().unwrap(),
                epsilon = 0.001
            );
        }
    }
}

#[cfg(test)]
mod test_problems {
    use optirustic_macros::test_with_retries;

    use crate::algorithms::{
        Algorithm, MaxGenerationValue, NSGA2Arg, StoppingConditionType, NSGA2,
    };
    use crate::core::builtin_problems::{
        SCHProblem, ZTD1Problem, ZTD2Problem, ZTD3Problem, ZTD4Problem,
    };
    use crate::core::test_utils::{check_exact_value, check_value_in_range};

    const BOUND_TOL: f64 = 1.0 / 1000.0;
    const LOOSE_BOUND_TOL: f64 = 0.1;

    #[test_with_retries(10)]
    /// Test problem 1 from Deb et al. (2002). Optional solution x in [0; 2]
    fn test_sch_problem() {
        let problem = SCHProblem::create().unwrap();
        let args = NSGA2Arg {
            number_of_individuals: 10,
            stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(1000)),
            crossover_operator_options: None,
            mutation_operator_options: None,
            parallel: Some(false),
            export_history: None,
            resume_from_file: None,
            seed: Some(10),
        };
        let mut algo = NSGA2::new(problem, args).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        // increase tolerance
        let bounds = -0.1..2.1;
        let invalid_x = check_value_in_range(&results.get_real_variables("x").unwrap(), &bounds);
        if !invalid_x.is_empty() {
            panic!("Some variables are outside the bounds: {:?}", invalid_x);
        }
    }

    #[test_with_retries(10)]
    /// Test the ZTD1 problem from Deb et al. (2002) with 30 variables. Solution x1 in [0; 1] and
    /// x2 to x30 = 0. The exact solutions are tested using a strict and loose bounds.
    fn test_ztd1_problem() {
        let number_of_individuals: usize = 30;
        let problem = ZTD1Problem::create(number_of_individuals).unwrap();
        let args = NSGA2Arg {
            number_of_individuals,
            stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(2500)),
            crossover_operator_options: None,
            mutation_operator_options: None,
            parallel: Some(false),
            export_history: None,
            resume_from_file: None,
            seed: Some(1),
        };
        let mut algo = NSGA2::new(problem, args).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        let x_bounds = 0.0 - BOUND_TOL..1.0 + BOUND_TOL;
        let invalid_x1 =
            check_value_in_range(&results.get_real_variables("x1").unwrap(), &x_bounds);
        if !invalid_x1.is_empty() {
            panic!("Some X1 variables are outside the bounds: {:?}", invalid_x1);
        }

        let x_bounds = -BOUND_TOL..BOUND_TOL;
        let x_other_bounds = -LOOSE_BOUND_TOL..LOOSE_BOUND_TOL;
        for xi in 2..=number_of_individuals {
            let var_values = results
                .get_real_variables(format!("x{xi}").as_str())
                .unwrap();
            let (x_other_outside_bounds, breached_range, b_type) =
                check_exact_value(&var_values, &x_bounds, &x_other_bounds, 5);
            if !x_other_outside_bounds.is_empty() {
                panic!(
                    "Found {} X2 to X30 solutions ({:?}) outside the {} bounds {:?}",
                    x_other_outside_bounds.len(),
                    x_other_outside_bounds,
                    b_type,
                    breached_range
                );
            }
        }
    }

    #[test_with_retries(10)]
    /// Test the ZTD2 problem from Deb et al. (2002) with 30 variables. Solution x1 in [0; 1] and
    /// x2 to x30 = 0. The exact solutions are tested using a strict and loose bounds.
    fn test_ztd2_problem() {
        let number_of_individuals: usize = 30;
        let problem = ZTD2Problem::create(number_of_individuals).unwrap();
        let args = NSGA2Arg {
            number_of_individuals,
            stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(2500)),
            crossover_operator_options: None,
            mutation_operator_options: None,
            parallel: Some(false),
            export_history: None,
            resume_from_file: None,
            seed: Some(1),
        };
        let mut algo = NSGA2::new(problem, args).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        let x_bounds = 0.0 - BOUND_TOL..1.0 + BOUND_TOL;
        let invalid_x1 =
            check_value_in_range(&results.get_real_variables("x1").unwrap(), &x_bounds);
        if !invalid_x1.is_empty() {
            panic!(
                "Found {} X1 variables outside the bounds {:?}",
                invalid_x1.len(),
                invalid_x1
            );
        }

        let x_bounds = -BOUND_TOL..BOUND_TOL;
        let x_other_bounds = -LOOSE_BOUND_TOL..LOOSE_BOUND_TOL;
        for xi in 2..=number_of_individuals {
            let var_name = format!("x{xi}");
            let var_values = results.get_real_variables(&var_name).unwrap();

            let (x_other_outside_bounds, breached_range, b_type) =
                check_exact_value(&var_values, &x_bounds, &x_other_bounds, 3);
            if !x_other_outside_bounds.is_empty() {
                panic!(
                    "Found {} {} solutions ({:?}) outside the {} bounds {:?}",
                    x_other_outside_bounds.len(),
                    var_name,
                    x_other_outside_bounds,
                    b_type,
                    breached_range
                );
            }
        }
    }

    #[test_with_retries(10)]
    /// Test the ZTD3 problem from Deb et al. (2002) with 30 variables. Solution x1 in [0; 1] and
    /// x2 to x30 = 0. The exact solutions are tested using a strict and loose bounds.
    fn test_ztd3_problem() {
        let number_of_individuals: usize = 30;
        let problem = ZTD3Problem::create(number_of_individuals).unwrap();
        let args = NSGA2Arg {
            number_of_individuals,
            stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(2500)),
            crossover_operator_options: None,
            mutation_operator_options: None,
            parallel: Some(false),
            export_history: None,
            resume_from_file: None,
            seed: Some(1),
        };
        let mut algo = NSGA2::new(problem, args).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        let x_bounds = 0.0 - BOUND_TOL..1.0 + BOUND_TOL;
        let invalid_x1 =
            check_value_in_range(&results.get_real_variables("x1").unwrap(), &x_bounds);
        if !invalid_x1.is_empty() {
            panic!(
                "Found {} X1 variables outside the bounds {:?}",
                invalid_x1.len(),
                invalid_x1
            );
        }

        let x_bounds = -BOUND_TOL..BOUND_TOL;
        let x_other_bounds = -LOOSE_BOUND_TOL..LOOSE_BOUND_TOL;
        for xi in 2..=number_of_individuals {
            let var_name = format!("x{xi}");
            let var_values = results.get_real_variables(&var_name).unwrap();

            let (x_other_outside_bounds, breached_range, b_type) =
                check_exact_value(&var_values, &x_bounds, &x_other_bounds, 3);
            if !x_other_outside_bounds.is_empty() {
                panic!(
                    "Found {} {} solutions ({:?}) outside the {} bounds {:?}",
                    x_other_outside_bounds.len(),
                    var_name,
                    x_other_outside_bounds,
                    b_type,
                    breached_range
                );
            }
        }
    }

    #[test_with_retries(10)]
    /// Test the ZTD4 problem from Deb et al. (2002) with 30 variables. Solution x1 in [0; 1] and
    /// x2 to x10 = 0. The exact solutions are tested using a strict and loose bounds.
    fn test_ztd4_problem() {
        let number_of_individuals: usize = 10;
        let args = NSGA2Arg {
            number_of_individuals,
            stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(3000)),
            crossover_operator_options: None,
            mutation_operator_options: None,
            parallel: Some(false),
            export_history: None,
            resume_from_file: None,
            seed: Some(1),
        };
        let problem = ZTD4Problem::create(number_of_individuals).unwrap();
        let mut algo = NSGA2::new(problem, args.clone()).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        let x_bounds = 0.0 - BOUND_TOL..1.0 + BOUND_TOL;
        let invalid_x1 =
            check_value_in_range(&results.get_real_variables("x1").unwrap(), &x_bounds);
        if !invalid_x1.is_empty() {
            panic!(
                "Found {} X1 variables outside the bounds {:?}",
                invalid_x1.len(),
                invalid_x1
            );
        }

        // relax strict bounds O(2). The final solution is still acceptable.
        let x_bounds = -BOUND_TOL * 10.0..BOUND_TOL * 10.0;
        let x_other_bounds = -LOOSE_BOUND_TOL..LOOSE_BOUND_TOL;
        for xi in 2..=number_of_individuals {
            let var_name = format!("x{xi}");
            let var_values = results.get_real_variables(&var_name).unwrap();

            let (x_other_outside_bounds, breached_range, b_type) =
                check_exact_value(&var_values, &x_bounds, &x_other_bounds, 3);
            if !x_other_outside_bounds.is_empty() {
                panic!(
                    "Found {} {} solutions ({:?}) outside the {} bounds {:?}",
                    x_other_outside_bounds.len(),
                    var_name,
                    x_other_outside_bounds,
                    b_type,
                    breached_range
                );
            }
        }
    }

    #[test_with_retries(10)]
    /// Test the ZTD6 problem from Deb et al. (2002) with 30 variables. Solution x1 in [0; 1] and
    /// x2 to x10 = 0. The exact solutions are tested using a strict and loose bounds.
    fn test_ztd6_problem() {
        let number_of_individuals: usize = 10;
        let problem = ZTD4Problem::create(number_of_individuals).unwrap();
        let args = NSGA2Arg {
            number_of_individuals,
            stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(1000)),
            crossover_operator_options: None,
            mutation_operator_options: None,
            parallel: Some(false),
            export_history: None,
            resume_from_file: None,
            seed: Some(1),
        };
        let mut algo = NSGA2::new(problem, args).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        let x_bounds = 0.0 - BOUND_TOL..1.0 + BOUND_TOL;
        let invalid_x1 =
            check_value_in_range(&results.get_real_variables("x1").unwrap(), &x_bounds);
        if !invalid_x1.is_empty() {
            panic!(
                "Found {} X1 variables outside the bounds {:?}",
                invalid_x1.len(),
                invalid_x1
            );
        }

        // relax strict bounds O(2). The final solution is still acceptable.
        let x_bounds = -BOUND_TOL * 10.0..BOUND_TOL * 10.0;
        let x_other_bounds = -LOOSE_BOUND_TOL..LOOSE_BOUND_TOL;
        for xi in 2..=number_of_individuals {
            let var_name = format!("x{xi}");
            let var_values = results.get_real_variables(&var_name).unwrap();

            let (x_other_outside_bounds, breached_range, b_type) =
                check_exact_value(&var_values, &x_bounds, &x_other_bounds, 3);
            if !x_other_outside_bounds.is_empty() {
                panic!(
                    "Found {} {} solutions ({:?}) outside the {} bounds {:?}",
                    x_other_outside_bounds.len(),
                    var_name,
                    x_other_outside_bounds,
                    b_type,
                    breached_range
                );
            }
        }
    }
}
