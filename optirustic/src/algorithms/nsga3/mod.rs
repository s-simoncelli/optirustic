use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::ops::Rem;
use std::sync::Arc;
use std::time::Instant;

use log::{debug, info, warn};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::algorithms::{Algorithm, ExportHistory, NSGA2, StoppingConditionType};
use crate::algorithms::nsga3::associate::AssociateToRefPoint;
use crate::algorithms::nsga3::niching::Niching;
use crate::algorithms::nsga3::normalise::Normalise;
use crate::core::{DataValue, Individual, OError, Population, Problem};
use crate::core::utils::get_rng;
use crate::operators::{
    Crossover, Mutation, ParetoConstrainedDominance, PolynomialMutation, PolynomialMutationArgs,
    Selector, SimulatedBinaryCrossover, SimulatedBinaryCrossoverArgs, TournamentSelector,
};
use crate::utils::{DasDarren1998, fast_non_dominated_sort, NumberOfPartitions};

mod associate;
mod niching;
mod normalise;

/// The data key where the normalised objectives are stored for each [`Individual`].
const NORMALISED_OBJECTIVE_KEY: &str = "normalised_objectives";

/// The data key where the perpendicular distance to a reference point is stored for each [`Individual`].
const MIN_DISTANCE: &str = "distance";

/// The data key where the reference point with [`MIN_DISTANCE`] is stored for each [`Individual`].
const REF_POINT: &str = "reference_point";

/// The data key where the reference point index for [`REF_POINT`] is stored.
const REF_POINT_INDEX: &str = "reference_point_index";

/// The type for the number of individuals in the population.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Nsga3NumberOfIndividuals {
    /// The number of individuals are set equal to the number of reference points.
    EqualToReferencePointCount,
    /// Set a custom number of individuals. This must be larger than the number of reference points
    /// generated by setting the [`NSGA3Arg::number_of_partitions`].
    Custom(usize),
}

/// Input arguments for the NSGA3 algorithm.
#[derive(Serialize, Deserialize, Clone)]
pub struct NSGA3Arg {
    /// The number of individuals in the population.
    pub number_of_individuals: Nsga3NumberOfIndividuals,
    /// The number of partitions to use to calculate the reference points or weight.
    pub number_of_partitions: NumberOfPartitions,
    /// The options of the Simulated Binary Crossover (SBX) operator. This operator is used to
    /// generate new children by recombining the variables of parent solutions. This defaults to
    /// [`SimulatedBinaryCrossoverArgs::default()`].
    /// NOTE: it is advisable to use a large `distribution_index` to prevent the problem explained in
    /// Section IIa point #3 in the paper. With many objectives, "two distant parent solutions are
    /// likely to produce offspring solutions that are also distant from parents", which should be
    /// prevented.
    pub crossover_operator_options: Option<SimulatedBinaryCrossoverArgs>,
    /// The options to Polynomial Mutation (PM) operator used to mutate the variables of an
    /// individual. This defaults to [`PolynomialMutationArgs::default()`],
    /// with a distribution index or index parameter of `20` and variable probability equal `1`
    /// divided by the number of real variables in the problem (i.e., each variable will have the
    /// same probability of being mutated).
    pub mutation_operator_options: Option<PolynomialMutationArgs>,
    /// The condition to use when to terminate the algorithm.
    pub stopping_condition: StoppingConditionType,
    /// Whether the objective and constraint evaluation in [`Problem::evaluator`] should run
    /// using threads. If the evaluation function takes a long time to run and return the updated
    /// values, it is advisable to set this to `true`. This defaults to `true`.
    pub parallel: Option<bool>,
    /// The options to configure the individual's history export. When provided, the algorithm will
    /// save objectives, constraints and solutions to a file each time the generation increases by
    /// a given step. This is useful to track convergence and inspect an algorithm evolution.
    pub export_history: Option<ExportHistory>,
    /// The seed used in the random number generator (RNG). You can specify a seed in case you want
    /// to try to reproduce results. NSGA2 is a stochastic algorithm that relies on an RNG at
    /// different steps (when population is initially generated, during selection, crossover and
    /// mutation) and, as such, may lead to slightly different solutions. The seed is randomly
    /// picked if this is `None`.
    pub seed: Option<u64>,
}

/// The Non-dominated Sorting Genetic Algorithm (NSGA3).
///
/// Implemented based on:
/// > K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using
/// Reference-Point-Based Non-dominated Sorting Approach, Part I: Solving Problems With Box
/// Constraints," in IEEE Transactions on Evolutionary Computation, vol. 18, no. 4, pp. 577-601,
/// Aug. 2014, doi: 10.1109/TEVC.2013.2281535
///
/// See: <https://10.1109/TEVC.2013.2281535>.
pub struct NSGA3 {
    /// The number of individuals to use in the population.
    number_of_individuals: usize,
    /// The vector of reference points
    reference_points: Vec<Vec<f64>>,
    /// The ideal point coordinates when the algorithm starts up to the current evolution
    ideal_point: Vec<f64>,
    /// The population with the solutions.
    population: Population,
    /// The problem being solved.
    problem: Arc<Problem>,
    /// The operator to use to select the individuals for reproduction. This is a binary tournament
    /// selector ([`TournamentSelector`]) with the [`ParetoConstrainedDominance`] comparison operator.
    selector_operator: TournamentSelector<ParetoConstrainedDominance>,
    /// The SBX operator to use to generate a new children by recombining the variables of parent
    /// solutions.
    crossover_operator: SimulatedBinaryCrossover,
    /// The PM operator to use to mutate the variables of an individual.
    mutation_operator: PolynomialMutation,
    /// The evolution step.
    generation: usize,
    /// The stopping condition.
    stopping_condition: StoppingConditionType,
    /// The time when the algorithm started.
    start_time: Instant,
    /// The configuration struct to export the algorithm history.
    export_history: Option<ExportHistory>,
    /// Whether the evaluation should run using threads
    parallel: bool,
    /// The seed to use.
    rng: Box<dyn RngCore>,
    /// The algorithm options
    args: NSGA3Arg,
}

impl Display for NSGA3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name().as_str())
    }
}

impl NSGA3 {
    /// Initialise the NSGA3 algorithm.
    ///
    /// # Arguments
    ///
    /// * `problem`: The problem being solved.
    /// * `args`: The [`NSGA3Arg`] arguments to customise the algorithm behaviour.
    ///
    /// returns: `NSGA3`.
    pub fn new(problem: Problem, options: NSGA3Arg) -> Result<Self, OError> {
        let name = "NSGA3".to_string();
        let nsga3_args = options.clone();

        let das_darren = DasDarren1998::new(
            problem.number_of_objectives(),
            &options.number_of_partitions,
        )?;
        let reference_points = das_darren.get_weights();
        info!(
            "Created {} reference directions",
            das_darren.number_of_points()
        );

        // create the population
        let mut number_of_individuals = match options.number_of_individuals {
            Nsga3NumberOfIndividuals::EqualToReferencePointCount => reference_points.len(),
            Nsga3NumberOfIndividuals::Custom(count) => {
                if count < 3 {
                    return Err(OError::AlgorithmInit(
                        name,
                        "The population size must have at least 3 individuals".to_string(),
                    ));
                }
                // add tolerance (for example Deb et al. sometimes uses pop + 1 = ref_point)
                if count - 1 > das_darren.number_of_points() as usize {
                    return Err(OError::AlgorithmInit(
                        name,
                        format!(
                            concat!(
                            "The number of individuals ({}) must be larger than the number of ",
                            "reference points ({}) to prevent unexpected behaviours. It is always ",
                            "suggested to associate at least one individual to a reference point"),
                            count,
                            das_darren.number_of_points()
                        ),
                    ));
                }
                count
            }
        };
        debug!("Population size set to {}", number_of_individuals);

        // force the population size as multiple of 2 so that the new number of generated offsprings
        // matches `number_of_individuals`
        if number_of_individuals.rem(2) != 0 {
            number_of_individuals -= 1;
            warn!(
                "The population size was reduced to {} so that it is a multiple of 2",
                number_of_individuals
            );
        }

        let problem = Arc::new(problem);
        let population = Population::init(problem.clone(), number_of_individuals);
        info!("Created initial random population");

        let selector_operator = TournamentSelector::<ParetoConstrainedDominance>::new(2);
        let mutation_options = match options.mutation_operator_options {
            Some(o) => o,
            None => PolynomialMutationArgs::default(problem.clone().as_ref()),
        };
        let mutation_operator = PolynomialMutation::new(mutation_options.clone())?;

        let crossover_options = options.crossover_operator_options.unwrap_or_default();
        let crossover_operator = SimulatedBinaryCrossover::new(crossover_options.clone())?;

        info!(
            "{}",
            NSGA2::algorithm_option_str(&problem, &crossover_options, &mutation_options)
        );

        Ok(Self {
            number_of_individuals,
            reference_points,
            ideal_point: vec![f64::INFINITY; problem.number_of_objectives()],
            population,
            problem,
            selector_operator,
            crossover_operator,
            mutation_operator,
            generation: 0,
            stopping_condition: options.stopping_condition,
            start_time: Instant::now(),
            parallel: options.parallel.unwrap_or(true),
            export_history: options.export_history,
            rng: get_rng(options.seed),
            args: nsga3_args,
        })
    }

    /// Get the normalised objective data stored in the `individual`.
    ///
    /// # Arguments
    ///
    /// * `individual`: The individual reference with the data.
    ///
    /// returns: `Result<DataValue, OError>`
    fn get_normalised_objectives(individual: &Individual) -> Result<DataValue, OError> {
        individual.get_data(NORMALISED_OBJECTIVE_KEY)
    }

    /// For each reference point count selected individuals in P_{t+1} associated with it. This
    /// returns `rho_j` which is a lookup map, mapping the reference point index to the number of
    /// linked individuals.
    ///
    /// # Arguments
    ///
    /// * `selected_individuals`: The individuals to use to count the association to the reference
    /// points.
    /// * `reference_points`: The reference points.
    ///
    /// returns: `Result<HashMap<usize, usize>, OError>`
    fn get_association_map(
        selected_individuals: &Population,
        reference_points: &[Vec<f64>],
    ) -> Result<HashMap<usize, usize>, OError> {
        let mut rho_j: HashMap<usize, usize> = HashMap::new();
        for ind in selected_individuals.individuals() {
            let ref_point_index = ind.get_data(REF_POINT_INDEX);
            match ref_point_index {
                Ok(index) => {
                    let index = index.as_usize()?;
                    rho_j.entry(index).and_modify(|v| *v += 1).or_insert(1);
                }
                Err(_) => continue,
            }
        }
        // fill the rest
        for ref_point_index in 0..reference_points.len() {
            rho_j.entry(ref_point_index).or_insert(0);
        }

        Ok(rho_j)
    }

    /// Get the reference points used in the evolution.
    ///
    /// return: `Vec<Vec<f64>>`
    pub fn reference_points(&self) -> Vec<Vec<f64>> {
        self.reference_points.clone()
    }
}

/// Implementation of Section IV of the paper.
impl Algorithm<NSGA3Arg> for NSGA3 {
    /// This assesses the initial random population.
    ///
    /// return: `Result<(), OError>`
    fn initialise(&mut self) -> Result<(), OError> {
        info!("Evaluating initial population");
        if self.parallel {
            NSGA3::do_parallel_evaluation(self.population.individuals_as_mut())?;
        } else {
            NSGA3::do_evaluation(self.population.individuals_as_mut())?;
        }

        info!("Initial evaluation completed");
        self.generation += 1;

        Ok(())
    }

    /// Evolve the population. The first part of this code comes from NSGA2::evolve(). NSGA3 mainly
    /// differs in the survival method.
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
            NSGA3::do_parallel_evaluation(self.population.individuals_as_mut())?;
        } else {
            NSGA3::do_evaluation(self.population.individuals_as_mut())?;
        }
        debug!("Evaluation done");

        debug!("Calculating fronts and ranks for new population");
        let sorting_results = fast_non_dominated_sort(self.population.individuals_as_mut(), false)?;
        debug!("Collected {} fronts", sorting_results.fronts.len());

        debug!("Selecting best individuals");
        // this is S_t in the paper, the population with the last front
        let mut new_population = Population::new();

        // Algorithm 1 in paper, step 5-7 - Fill the new population up to the last front.
        let mut last_front: Option<Vec<Individual>> = None;
        for (fi, front) in sorting_results.fronts.into_iter().enumerate() {
            if new_population.len() + front.len() <= self.number_of_individuals {
                // population does not overflow with new front
                debug!("Adding front #{} (size: {})", fi + 1, front.len());
                new_population.add_new_individuals(front);
            } else if new_population.len() == self.number_of_individuals {
                debug!("Population reached target size");
                break;
            } else {
                // Algorithm 1, step 12. Population is filled up to front l-1 (P_{t+1})
                debug!(
                    "Population almost full ({} individuals)",
                    new_population.len()
                );
                // this is F_l
                last_front = Some(front);
                break;
            }
        }

        // Algorithm 1, step 11-18 - Complete the population using the members from the last front.
        if let Some(last_front) = last_front {
            // store the last population index containing individuals up to front F_l
            let first_dom_index = new_population.len();
            let missing_item_count = self.number_of_individuals - new_population.len();
            debug!("{missing_item_count} must be added from the last front");

            // add the last_front F_l to create S_t
            new_population.add_new_individuals(last_front);

            // Algorithm 1, step 14 - Calculate f_n
            debug!("Normalising all individuals");
            let mut norm =
                Normalise::new(&mut self.ideal_point, new_population.individuals_as_mut())?;
            norm.calculate()?;

            // Algorithm 1, step 15
            debug!("Associating reference points to all individuals");
            let mut assoc = AssociateToRefPoint::new(
                new_population.individuals_as_mut(),
                &self.reference_points,
            )?;
            assoc.calculate()?;

            // Algorithm 1, step 16
            // re-split population in P_{t+1} (S_t without the last front) and individuals in front F_l
            let mut potential_individuals = new_population.drain(first_dom_index..);
            // rename variable for clarity
            let mut selected_individuals = new_population;

            // for each reference point count selected individuals in P_{t+1} associated with it.
            // rho_j is a lookup map mapping the reference point index to the number of linked
            // individuals
            let mut rho_j =
                NSGA3::get_association_map(&selected_individuals, &self.reference_points)?;

            // Algorithm 4 - Niching
            debug!("Niching");
            let mut n = Niching::new(
                &mut selected_individuals,
                &mut potential_individuals,
                missing_item_count,
                &mut rho_j,
                &mut self.rng,
            )?;
            n.calculate()?;

            // update the population
            self.population = selected_individuals;
        } else {
            // update the population
            self.population = new_population;
        }

        self.generation += 1;
        Ok(())
    }

    fn generation(&self) -> usize {
        self.generation
    }

    fn name(&self) -> String {
        "NSGA3".to_string()
    }

    fn start_time(&self) -> &Instant {
        &self.start_time
    }

    fn stopping_condition(&self) -> &StoppingConditionType {
        &self.stopping_condition
    }

    fn population(&self) -> &Population {
        &self.population
    }

    fn problem(&self) -> Arc<Problem> {
        self.problem.clone()
    }

    fn export_history(&self) -> Option<&ExportHistory> {
        self.export_history.as_ref()
    }

    fn additional_export_data(&self) -> Option<HashMap<String, DataValue>> {
        let mut data = HashMap::new();
        let mut points: Vec<DataValue> = Vec::new();
        for point in self.reference_points() {
            points.push(DataValue::Vector(point));
        }
        data.insert(
            "reference_points".to_string(),
            DataValue::DataVector(points),
        );
        data.insert(
            "ideal_point".to_string(),
            DataValue::Vector(self.ideal_point.clone()),
        );
        Some(data)
    }

    fn algorithm_options(&self) -> NSGA3Arg {
        self.args.clone()
    }
}

#[cfg(test)]
mod test_problems {
    use crate::algorithms::{
        Algorithm, MaxGeneration, NSGA3, NSGA3Arg, Nsga3NumberOfIndividuals, StoppingConditionType,
    };
    use crate::core::builtin_problems::dtlz1;
    use crate::core::test_utils::check_exact_value;
    use crate::operators::{PolynomialMutationArgs, SimulatedBinaryCrossoverArgs};
    use crate::utils::NumberOfPartitions;

    #[test]
    /// Test the ZTD1 problem from Deb et al. (2013) with M=3 (see Table III).
    fn test_ztd1_problem_1() {
        // see Table I
        let number_objectives: usize = 3;
        let k: usize = 5;
        let number_variables: usize = number_objectives + k - 1; // M + k - 1 with k = 5 (Section Va)
        let problem = dtlz1(number_variables, number_objectives).unwrap();
        // The number of partitions used in the paper when M=3 - Table I
        let number_of_partitions = NumberOfPartitions::OneLayer(12);

        // see Table II
        let crossover_operator_options = SimulatedBinaryCrossoverArgs {
            distribution_index: 30.0,
            crossover_probability: 1.0,
            variable_probability: 1.0,
        };
        // eta_m = 20 - probability  1/n_vars
        let mutation_operator_options = PolynomialMutationArgs::default(&problem);

        let args = NSGA3Arg {
            // see Table I
            number_of_individuals: Nsga3NumberOfIndividuals::Custom(92),
            number_of_partitions,
            crossover_operator_options: Some(crossover_operator_options),
            mutation_operator_options: Some(mutation_operator_options),
            // see Table III
            stopping_condition: StoppingConditionType::MaxGeneration(MaxGeneration(400)),
            parallel: None,
            export_history: None,
            seed: Some(1),
        };

        let mut algo = NSGA3::new(problem, args).unwrap();
        assert_eq!(algo.reference_points().len(), 91);

        algo.run().unwrap();
        let results = algo.get_results();

        // All objective points lie on the plane passing through the 0.5 intercept on each axis (i.e.
        // the sum of the objective coordinate is close to 0.5). Because of randomness a few solutions
        // may breach this condition.
        let obj_sum: Vec<f64> = results
            .individuals
            .iter()
            .map(|ind| ind.get_objective_values().unwrap().iter().sum())
            .collect();
        let strict_range = 0.47..0.53;
        let loose_range = -10.0..10.0;
        let (x_other_outside_bounds, breached_range, b_type) =
            check_exact_value(&obj_sum, &strict_range, &loose_range, 2);
        if !x_other_outside_bounds.is_empty() {
            panic!(
                "Found {} objectives ({:?}) outside the {} bounds {:?}",
                x_other_outside_bounds.len(),
                x_other_outside_bounds,
                b_type,
                breached_range
            );
        }
    }
}
