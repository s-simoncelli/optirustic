use std::ops::Rem;
use std::sync::Arc;

use chrono::{DateTime, Local};
use log::{debug, info};

use crate::algorithms::{Algorithm, StoppingConditionType};
use crate::core::{Individual, Individuals, OError, Population, Problem, VariableValue};
use crate::core::individual::IndividualsMut;
use crate::core::utils::{argsort, vector_max, vector_min};
use crate::operators::{
    BinaryComparisonOperator, Crossover, Mutation, ParetoConstrainedDominance, PolynomialMutation,
    PolynomialMutationArgs, PreferredSolution, Selector, SimulatedBinaryCrossover,
    SimulatedBinaryCrossoverArgs, TournamentSelector,
};
use crate::operators::comparison::CrowdedComparison;

/// Input arguments for the NSGA2 algorithm.
pub struct NSGA2Arg {
    /// The number of individuals to use in the population. This must be a multiple of 2.
    pub number_of_individuals: usize,
    /// The problem being solved.
    pub problem: Problem,
    /// The options of the Simulated Binary Crossover (SBX) operator. This operator is used to
    /// generate new children by recombining the variables of parent solutions. This defaults to
    /// `SimulatedBinaryCrossoverArgs::default()`.
    pub crossover_operator_options: Option<SimulatedBinaryCrossoverArgs>,
    /// The options to Polynomial Mutation (PM) operator used to mutate the variables of an
    /// individual. This defaults to `SimulatedBinaryCrossoverArgs::default()`, with a distribution
    /// index or index parameter of 20 and variable probability equal 1 divided by the number of
    /// real variables in the problem (i.e., each variable will have the same probability of being
    /// mutated).
    pub mutation_operator_options: Option<PolynomialMutationArgs>,
    /// The condition to use when to terminate the algorithm.
    pub stopping_condition: StoppingConditionType,
    /// Whether the objective and constraint evaluation in [`Problem::evaluator`] should run
    /// using threads. If the evaluation function takes a long time to run and return the updated
    /// values, it is advisable to set this to `true`. This defaults to `true`.
    pub parallel: Option<bool>,
}

/// The Non-dominated Sorting Genetic Algorithm (NSGA2).
///
/// Implemented based on:
/// > K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multi-objective genetic
/// > algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp.
/// > 182-197, April 2002, doi: 10.1109/4235.996017.
///
/// See: <https://doi.org/10.1109/4235.996017>.
pub struct NSGA2 {
    /// The number of individuals to use in the population.
    number_of_individuals: usize,
    /// The population with the solutions.
    population: Population,
    /// The problem being solved.
    problem: Arc<Problem>,
    /// The operator to use to select the individuals for reproduction.
    selector_operator: TournamentSelector<CrowdedComparison>,
    /// The operator to use to generate a new children by recombining the variables of parent
    /// solutions. This is a binary tournament selector ([`TournamentSelector`]) with the
    /// [`CrowdedComparison`] comparison operator.
    crossover_operator: SimulatedBinaryCrossover,
    /// The operator to use to mutate the variables of an individual.
    mutation_operator: PolynomialMutation,
    /// The evolution step.
    generation: usize,
    /// The stopping condition.
    stopping_condition: StoppingConditionType,
    /// The time when the algorithm started.
    start_time: DateTime<Local>,
    /// Whether the evaluation should run using threads
    parallel: bool,
}

impl NSGA2 {
    /// Initialise the NSGA2 algorithm.
    ///
    /// # Arguments
    ///
    /// * `args`: The [`NSGA2Arg`] input arguments.
    ///
    /// returns: `NSGA2`.
    pub fn new(args: NSGA2Arg) -> Result<Self, OError> {
        if args.number_of_individuals < 3 {
            return Err(OError::AlgorithmInit(
                "NSGA2".to_string(),
                "The population size must have at least 3 individuals".to_string(),
            ));
        }
        // force the population size as multiple of 2 so that the new number of generated offsprings
        // matches `number_of_individuals`
        if args.number_of_individuals.rem(2) != 0 {
            return Err(OError::AlgorithmInit(
                "NSGA2".to_string(),
                "The population size must be a multiple of 2".to_string(),
            ));
        }

        let problem = Arc::new(args.problem);
        info!("Created initial random population");
        let population = Population::init(problem.clone(), args.number_of_individuals);

        let mutation_options = match args.mutation_operator_options {
            Some(o) => o,
            None => PolynomialMutationArgs::default(problem.clone().as_ref()),
        };
        let mutation_operator = PolynomialMutation::new(mutation_options.clone())?;

        let crossover_options = args.crossover_operator_options.unwrap_or_default();
        let crossover_operator = SimulatedBinaryCrossover::new(crossover_options.clone())?;

        // log options
        let mut options: String = "Algorithm options are:\n".to_owned();
        options.push_str(
            format!("\t* Number of variables {:>13}\n\t* Number of objectives {:>12}\n\t* Number of constraints {:>11}\n",
                    problem.number_of_variables(),
                    problem.number_of_objectives(),
                    problem.number_of_constraints()
            ).as_str()
        );
        options.push_str(
            format!(
                "\t* Crossover distribution index {:>5}\n\t* Crossover probability {:>11}\n\t* Crossover var probability {:>9}\n",
                crossover_options.distribution_index, crossover_options.crossover_probability, crossover_options.variable_probability,
            )
            .as_str(),
        );
        options.push_str(
            format!(
                "\t* Mutation index parameter {:>9}\n\t* Mutation var probability {:>10}",
                mutation_options.index_parameter, crossover_options.variable_probability,
            )
            .as_str(),
        );
        info!("{}", options);

        Ok(Self {
            number_of_individuals: args.number_of_individuals,
            problem,
            population,
            selector_operator: TournamentSelector::<CrowdedComparison>::new(2),
            crossover_operator,
            mutation_operator,
            generation: 0,
            stopping_condition: args.stopping_condition,
            start_time: Local::now(),
            parallel: args.parallel.unwrap_or(true),
        })
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
    ) -> Result<crate::algorithms::utils::NonDominatedSortResults, OError> {
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
        let mut dominated_solutions: Vec<Vec<usize>> =
            individuals.iter().map(|_| Vec::new()).collect();
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

        Ok(crate::algorithms::utils::NonDominatedSortResults {
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
            debug!("Setting crowding distance to Inf for all individuals. At least 3 individuals are needed");

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
                debug!("Setting crowding distance to Inf for all individuals. The min/max range is too small");
                return Ok(());
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
                // get the corresponding individual to sorted objective
                let ind_i = sorted_idx[obj_i];
                let current_distance = individuals
                    .individual(ind_i)?
                    .get_data(data_name)
                    .unwrap_or(VariableValue::Real(0.0));

                if let VariableValue::Real(current_distance) = current_distance {
                    let delta = (obj_values[obj_i + 1] - obj_values[obj_i - 1]) / delta_range;
                    individuals
                        .individual_as_mut(ind_i)?
                        .set_data(data_name, VariableValue::Real(current_distance + delta));
                }
            }
        }

        Ok(())
    }
}

/// Implementation of Section IIIC of the paper.
impl Algorithm for NSGA2 {
    /// This assesses the initial random population and sets the individual's ranks and crowding
    /// distance needed in [`self.evolve`].
    ///
    /// return: `Result<(), OError>`
    fn initialise(&mut self) -> Result<(), OError> {
        info!("Evaluating initial population");
        if self.parallel {
            NSGA2::do_parallel_evaluation(self.population.individuals_as_mut())?;
        } else {
            NSGA2::do_evaluation(self.population.individuals_as_mut())?;
        }

        info!("Initial evaluation completed");
        self.generation += 1;

        Ok(())
    }

    fn evolve(&mut self) -> Result<(), OError> {
        debug!("Calculating rank");
        NSGA2::fast_non_dominated_sort(self.population.individuals_as_mut())?;

        debug!("Calculating crowding distance");
        NSGA2::set_crowding_distance(self.population.individuals_as_mut())?;

        // Create the new population, based on the population at the previous time-step, of size
        // `self.number_of_individuals`. The loop will add two individuals at the time.
        debug!("Generating new population (selection+mutation)");
        let mut offsprings: Vec<Individual> = Vec::new();
        for _ in 0..self.number_of_individuals / 2 {
            let parents = self
                .selector_operator
                .select(self.population.individuals(), 2)?;

            // generate the 2 children with crossover
            let children = self
                .crossover_operator
                .generate_offsprings(&parents[0], &parents[1])?;

            // mutate them
            offsprings.push(self.mutation_operator.mutate_offsprings(&children.child1)?);
            offsprings.push(self.mutation_operator.mutate_offsprings(&children.child2)?);
        }
        debug!("Combining parents and offsprings in new population");
        self.population.add_new_individuals(offsprings);
        debug!("New population size is {}", self.population.size());

        debug!("Evaluating population");
        if self.parallel {
            NSGA2::do_parallel_evaluation(self.population.individuals_as_mut())?;
        } else {
            NSGA2::do_evaluation(self.population.individuals_as_mut())?;
        }
        debug!("Evaluation done");

        debug!("Calculating fronts and ranks for new population");
        let sorting_results = NSGA2::fast_non_dominated_sort(self.population.individuals_as_mut())?;
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
            if new_population.size() + front.len() <= self.number_of_individuals {
                debug!("Adding front #{} (size: {})", fi + 1, front.len());
                new_population.add_new_individuals(front);
            } else if new_population.size() == self.number_of_individuals {
                debug!("Population reached target size");
                break;
            } else {
                debug!(
                    "Population almost full ({} individuals)",
                    new_population.size()
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
                i.get_data("crowding_distance")
                    .unwrap()
                    .as_real()
                    .unwrap()
                    .total_cmp(&o.get_data("crowding_distance").unwrap().as_real().unwrap())
            });
            last_front.reverse();

            // add the items ti complete the population
            last_front.truncate(self.number_of_individuals - new_population.size());
            new_population.add_new_individuals(last_front);
        }

        // update the population
        self.population = new_population;

        self.generation += 1;
        Ok(())
    }

    fn generation(&self) -> usize {
        self.generation
    }

    fn name(&self) -> String {
        "NSGA2".to_string()
    }

    fn start_time(&self) -> DateTime<Local> {
        self.start_time
    }

    fn stopping_condition(&self) -> StoppingConditionType {
        self.stopping_condition.clone()
    }

    fn population(&self) -> Population {
        self.population.clone()
    }

    fn problem(&self) -> Arc<Problem> {
        self.problem.clone()
    }
}

/// Outputs of the non-dominated sort algorithm.
pub struct NonDominatedSortResults {
    /// A vector containing sub-vectors. Each child vector represents a front (with the first being
    /// the primary non-dominated front with solutions of rank 1); each child vector contains
    /// the individuals belonging to that front.
    pub fronts: Vec<Vec<Individual>>,
    /// This is [`crate::algorithms::utils::NonDominatedSortResults::fronts`], but the individuals are given as indexes
    /// instead of references. Each index refers to the vector of individuals passed to
    /// [`fast_non_dominated_sort`].
    pub front_indexes: Vec<Vec<usize>>,
    /// Number of individuals that dominates a solution at a given vector index. When the counter
    /// is 0, the solution is non-dominated. This is `n_p` in the paper.
    pub domination_counter: Vec<usize>,
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use float_cmp::assert_approx_eq;

    use crate::algorithms::NSGA2;
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
        let result = NSGA2::fast_non_dominated_sort(&mut individuals).unwrap();

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
        NSGA2::set_crowding_distance(&mut individuals).unwrap();
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
        let result = NSGA2::fast_non_dominated_sort(&mut individuals).unwrap();

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
    fn test_crowding_distance_not_enough_points() {
        let objectives = vec![[0.0, 0.0], [50.0, 50.0]];
        let mut individuals = create_individuals(objectives);
        NSGA2::set_crowding_distance(&mut individuals).unwrap();
        for i in individuals {
            assert_eq!(
                i.get_data("crowding_distance").unwrap(),
                VariableValue::Real(f64::INFINITY)
            );
        }
    }

    #[test]
    /// Test the crowding distance algorithm (min and max of objective is equal).
    fn test_crowding_distance_min_max_range() {
        let objectives = vec![[10.0, 20.0], [10.0, 20.0], [10.0, 20.0], [10.0, 20.0]];
        let mut individuals = create_individuals(objectives);
        NSGA2::set_crowding_distance(&mut individuals).unwrap();
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
            NSGA2::set_crowding_distance(&mut individuals).unwrap();

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
        NSGA2::set_crowding_distance(&mut individuals).unwrap();

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
        NSGA2::set_crowding_distance(&mut individuals).unwrap();

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
        NSGA2::set_crowding_distance(&mut individuals).unwrap();

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
