use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::Instant;

use log::{debug, info};
use rand::prelude::SliceRandom;
use rand::RngCore;
use serde::Serialize;

use crate::algorithms::{Algorithm, ExportHistory, NSGA2, StoppingConditionType};
use crate::core::{DataValue, Individual, Individuals, OError, Population, Problem};
use crate::core::utils::get_rng;
use crate::operators::{
    Crossover, CrowdedComparison, Mutation, PolynomialMutation, PolynomialMutationArgs, Selector,
    SimulatedBinaryCrossover, SimulatedBinaryCrossoverArgs, TournamentSelector,
};
use crate::utils::{
    argmin, argmin_by, DasDarren1998, fast_non_dominated_sort, perpendicular_distance,
    solve_linear_system, vector_max, vector_min,
};

/// The data key where the normalised objectives are stored for each [`Individual`].
const NORMALISED_OBJECTIVE_KEY: &str = "normalised_objectives";

/// The data key where the perpendicular distance to a reference point is stored for each [`Individual`].
const MIN_DISTANCE: &str = "distance";

/// The data key where the reference point with [`MIN_DISTANCE`] is stored for each [`Individual`].
const REF_POINT: &str = "reference_point";

/// The data key where the reference point index for [`REF_POINT`] is stored.
const REF_POINT_INDEX: &str = "reference_point_index";

/// Input arguments for the NSGA3 algorithm.
#[derive(Serialize, Clone)]
pub struct NSGA3Arg {
    /// The number of individuals in the population. This must be larger than the number of
    /// reference points generated by setting the [`NSGA3Arg::number_of_partitions`].
    pub number_of_individuals: usize,
    /// The number of uniform gaps between two consecutive points along all objective axis on
    /// the hyperplane. This is used to calculate the reference points or weight using the
    /// [`DasDarren1998`] approach.
    pub number_of_partitions: usize,
    /// The options of the Simulated Binary Crossover (SBX) operator. This operator is used to
    /// generate new children by recombining the variables of parent solutions. This defaults to
    /// [`SimulatedBinaryCrossoverArgs::default()`].
    pub crossover_operator_options: Option<SimulatedBinaryCrossoverArgs>,
    /// The options to Polynomial Mutation (PM) operator used to mutate the variables of an
    /// individual. This defaults to [`SimulatedBinaryCrossoverArgs::default()`],
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
    /// to try to reproduce results. NSGA2 is a stochastic algorithm that relies on a RNG at
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
        if options.number_of_individuals < 3 {
            return Err(OError::AlgorithmInit(
                name,
                "The population size must have at least 3 individuals".to_string(),
            ));
        }
        if options.number_of_partitions < 3 {
            return Err(OError::AlgorithmInit(
                name,
                "The number of partition must be at least 3".to_string(),
            ));
        }

        let nsga3_args = options.clone();
        let problem = Arc::new(problem);
        let population = Population::init(problem.clone(), options.number_of_partitions);
        info!("Created initial random population");

        let das_darren =
            DasDarren1998::new(problem.number_of_objectives(), options.number_of_partitions);
        let reference_points = das_darren.get_weights();

        if options.number_of_individuals > das_darren.number_of_points() as usize {
            return Err(OError::AlgorithmInit(
                name,
                format!(
                    concat!(
                "The number of individuals ({}) must be larger than the number of reference ",
                    "points ({}) to prevent unexpected behaviours. It is always advisable to ",
                    "associate at least one individual to a reference point"),
                    options.number_of_individuals,
                    das_darren.number_of_points()
                ),
            ));
        }
        info!("Created reference directions");

        let selector_operator = TournamentSelector::<CrowdedComparison>::new(2);
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
            number_of_individuals: options.number_of_individuals,
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

    /// Get the normalised objective data stored in the `individual`
    ///
    /// # Arguments
    ///
    /// * `individual`: The individual reference with the data.
    ///
    /// returns: `Result<DataValue, OError>`
    fn get_normalised_objectives(individual: &Individual) -> Result<DataValue, OError> {
        individual.get_data(NORMALISED_OBJECTIVE_KEY)
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
    /// differs in the selection method.
    fn evolve(&mut self) -> Result<(), OError> {
        // Create the new population, based on the population at the previous time-step, of size
        // self.number_of_individuals. The loop adds two individuals at the time.
        debug!("Generating new population (selection+mutation)");
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
                last_front = Some(front.clone());
                break;
            }
        }

        // Algorithm 1, step 11-18 - Complete the population using the members from the last front.
        if let Some(last_front) = last_front {
            // store the last population index containing individuals up to front F_l{-1}
            let last_nd_index = new_population.len();
            let missing_item_count = self.number_of_individuals - new_population.len();
            // add the last_front F_l to create S_t
            new_population.add_new_individuals(last_front);

            // Algorithm 1, step 14 - Calculate f_n
            debug!("Normalising individuals");
            let mut norm =
                Normalise::new(&mut self.ideal_point, new_population.individuals_as_mut())?;
            norm.calculate()?;

            // Algorithm 1, step 15
            debug!("Associating reference points to individuals");
            let mut assoc = AssociateToRefPoint::new(
                new_population.individuals_as_mut(),
                &self.reference_points,
            )?;
            assoc.calculate()?;

            // Algorithm 1, step 16
            // re-split population in P_{t+1} (S_t without the last front) and individuals in front F_l
            let mut potential_individuals = new_population.drain(last_nd_index..);

            // for each reference point count selected individuals in P_{t+1} associated with it.
            // rho_j is a lookup map mapping the reference point index to the number of linked
            // individuals in P_{t+1}
            let mut rho_j: HashMap<usize, usize> = HashMap::new();
            for ind in new_population.0.iter() {
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
            for ref_point_index in 0..self.reference_points.len() {
                rho_j.entry(ref_point_index).or_insert(0);
            }

            // Algorithm 4 - Niching
            debug!("Niching");
            let mut n = Niching::new(
                &mut new_population,
                &mut potential_individuals,
                missing_item_count,
                &mut rho_j,
                &mut self.rng,
            );
            n.calculate()?;
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

    fn algorithm_options(&self) -> &NSGA3Arg {
        &self.args
    }
}

/// This implements "Algorithm 2" in the paper which normalises the population members using the
/// adaptive ideal point and the intercepts of the hyper-plane passing through the extreme points
/// and crossing the objective space axis. Steps 8-10 are ignored because [`NSGA3`] implementation
/// directly uses Das and Darren's approach with already-normalised points.
///
/// This procedure:
///  - updates the ideal point. The new coordinates may differ from the original point if any
///    objective, calculated at the current evolution, is lower than the one at the previous
///    evolution.
///  - store the normalised objective in the objective space, with respect to the new ideal point
///    and hyper-plane intercepts, in the "translated_objective" data key in each [`Individual`].
///    To retrieve those, use [`NSGA3::get_normalised_objectives()`].
struct Normalise<'a> {
    /// The coordinate of the ideal point from the previous evolution.
    ideal_point: &'a mut Vec<f64>,
    /// The individual that needs normalisation.
    individuals: &'a mut [Individual],
    /// The name of this algorithm.
    name: String,
}

impl<'a> Normalise<'a> {
    /// Build the [`Normalise`] struct.
    ///
    /// # Arguments
    ///
    /// * `ideal_point`: The coordinate of the ideal point  from the previous evolution.
    /// * `individuals`: The individual that needs normalisation.
    ///
    /// returns: `Result<Normalise, OError>`
    fn new(
        ideal_point: &'a mut Vec<f64>,
        individuals: &'a mut [Individual],
    ) -> Result<Self, OError> {
        let name = "NSGA3-Normalise".to_string();
        if individuals.is_empty() {
            return Err(OError::AlgorithmRun(
                name,
                "The vector of individuals is empty".to_string(),
            ));
        }
        Ok(Normalise {
            name,
            ideal_point,
            individuals,
        })
    }

    /// Normalise the population members using "Algorithm 2" from the paper. Objectives are first
    /// translated with respect to the new ideal point and then scaled using the intercepts of the
    /// linear hyper-plane passing through the extreme points.
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals with the objectives
    ///
    /// returns: `()`: this only updates the ideal point and stores the normalised objective in the
    /// [`Individual`]'s data.
    pub fn calculate(&mut self) -> Result<(), OError> {
        // Step 2 - calculate the new ideal point, based paragraph IV-C, as the minimum value for
        // each objective from the start of the algorithm evolution up to the current evolution
        // step.
        let problem = self.individuals.first().unwrap().problem();
        for (j, obj_name) in problem.objective_names().iter().enumerate() {
            let new_min = vector_min(&self.individuals.objective_values(obj_name)?)?;
            // update the point if its coordinate is smaller
            if new_min < self.ideal_point[j] {
                self.ideal_point[j] = new_min;
            }
        }

        // Step 3 - Translate the individuals' objectives with respect to the ideal point. This
        // implements the calculation of `f'_j(x)` in section IV-C of the paper.
        for x in self.individuals.iter_mut() {
            let translated_objectives = x
                .get_objective_values()?
                .iter()
                .enumerate()
                .map(|(j, v)| v - self.ideal_point[j])
                .collect();
            x.set_data(
                NORMALISED_OBJECTIVE_KEY,
                DataValue::Vector(translated_objectives),
            );
        }

        // Step 4 - Calculate the vector of extreme points
        let mut extreme_points = vec![];
        for j in 0..problem.number_of_objectives() {
            // Extreme point z_j_max for current objective
            let mut weights = vec![10.0_f64.powi(-6); problem.number_of_objectives()];
            weights[j] = 1.0;

            let mut min_value = f64::INFINITY; // minimum ASF
            let mut ind_index = 0; // index of individual with minimum ASF

            for (x_idx, ind) in self.individuals.iter().enumerate() {
                let f_j = NSGA3::get_normalised_objectives(ind)?;
                let value = self.asf(f_j.as_vec()?, &weights)?;
                if value < min_value {
                    min_value = value;
                    ind_index = x_idx;
                }
            }
            extreme_points.push(
                NSGA3::get_normalised_objectives(&self.individuals[ind_index])?
                    .as_vec()?
                    .clone(),
            );
        }

        // Step 6 - Compute intercepts a_j with the least-square method
        let intercept_result = Self::calculate_plane_intercepts(&extreme_points)
            .map_err(|e| OError::AlgorithmRun(self.name.clone(), e));
        let intercepts: Vec<f64> = match intercept_result? {
            None => {
                // no solution found or intercepts are too small - get worst (max) for each objective
                let mut max_points = vec![];
                for j in 0..problem.number_of_objectives() {
                    let mut obj_j_values = Vec::new();
                    for ind in self.individuals.iter_mut() {
                        obj_j_values.push(NSGA3::get_normalised_objectives(ind)?.as_vec()?[j]);
                    }
                    obj_j_values.push(f64::EPSILON);
                    max_points.push(vector_max(&obj_j_values)?);
                }
                max_points
            }
            Some(i) => i,
        };

        // Step 7 - Normalize objectives (f_n). The denominator differs from Eq. 5 in the paper
        // because the intercepts are already calculated using the translated objectives. The new
        // values are updated for all individuals.
        for individual in self.individuals.iter_mut() {
            let tmp: Vec<f64> = NSGA3::get_normalised_objectives(individual)?
                .as_vec()?
                .iter()
                .enumerate()
                .map(|(oi, obj_value)| obj_value / intercepts[oi])
                .collect();
            individual.set_data(NORMALISED_OBJECTIVE_KEY, DataValue::Vector(tmp));
        }
        Ok(())
    }

    /// Use the least square method to calculate the coefficients of the equation of the plane
    /// passing through the vector of `points`. For example, for a 3D system the equation being
    /// used is: $ax + by + cz = 1$. The coefficient vector $x = [a, b, c]$ is found by solving
    /// the linear system $A \cdot x = b$ where `A` is
    ///
    ///          | x_0   y_0   z_0 |
    ///      A = | x_1   y_1   z_1 |
    ///          |       ...       |
    ///          | x_n   y_n   z_n |
    /// `n` the size of `points` and $b = [1, 1, 1]$. The intercepts are then calculated as the
    /// inverse of `x` as $1/x$. For example for the z-axis intercept (with x=0 and y=0), the point
    /// is found by solving $cz = 1$ or $1/x[2]$.
    ///
    /// # Arguments
    ///
    /// * `points`: The point coordinates passing through the plane to calculate.
    ///
    /// returns: `Result<Vec<f64>, OError>`: The $ a_i $ intercept values for each axis (see Fig.2
    /// in the paper) or `None` if the intercepts are close to `0`.
    fn calculate_plane_intercepts(points: &[Vec<f64>]) -> Result<Option<Vec<f64>>, String> {
        let b = vec![1.0; points.len()];
        let plane_coefficients = solve_linear_system(points, &b, false)?;
        let intercepts: Vec<f64> = plane_coefficients.iter().map(|v| 1.0 / v).collect();

        // check that the intercepts are above a threshold
        if intercepts.iter().all(|v| *v >= 10_f64.powi(-6)) {
            Ok(Some(intercepts))
        } else {
            Ok(None)
        }
    }

    /// Calculate the achievement scalarising function with weight vector `w`. This is Eq. 4 in the
    /// paper.
    ///
    /// # Arguments
    ///
    /// * `translated_objective`: The translated objective for an individual. This is f'_j(x).
    /// * `weights`: The weight vector.
    ///
    /// returns: `Result<Vec<f64>, OError>`
    fn asf(&self, translated_objective: &[f64], weights: &Vec<f64>) -> Result<f64, OError> {
        let asf: Vec<f64> = translated_objective
            .iter()
            .zip(weights)
            .map(|(x, w)| x / w)
            .collect();
        vector_max(&asf)
    }
}

/// This implements "Algorithm 3" in the paper which associates each individual's normalised
/// objectives to a reference point.
struct AssociateToRefPoint<'a> {
    /// The individuals containing the normalised objectives.
    individuals: &'a mut [Individual],
    /// The reference points
    reference_points: &'a [Vec<f64>],
}

impl<'a> AssociateToRefPoint<'a> {
    /// Build the [`AssociateToRefPoint`] structure. This returns an error if the reference point
    /// or individual's normalised objectives are not between 0 and 1.
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals containing the normalised objectives.
    /// * `reference_points`: The reference points to associate the objectives to.
    ///
    /// returns: `Result<Self, OError>`
    fn new(
        individuals: &'a mut [Individual],
        reference_points: &'a [Vec<f64>],
    ) -> Result<Self, OError> {
        // check vector sizes
        for (i, ind) in individuals.iter().enumerate() {
            Self::check_bounds(
                NSGA3::get_normalised_objectives(ind)?.as_vec()?,
                format!("normalised objective vector #{}", i + 1),
            )?;
        }
        for (ri, point) in reference_points.iter().enumerate() {
            Self::check_bounds(point, format!("reference point vector #{}", ri + 1))?;
        }
        Ok(Self {
            individuals,
            reference_points,
        })
    }

    /// Associate the individuals to a reference point. If an association is found, this function
    /// stores the distance, the reference point coordinates and reference point index of
    /// [`self.reference_points`] in the individual's data.
    ///
    /// return `Result<(), OError>`
    fn calculate(&mut self) -> Result<(), OError> {
        // steps 1-3 are skipped because `reference_points` are already normalised

        // step 4-7
        for ind in self.individuals.iter_mut() {
            // fetch the data
            let data = NSGA3::get_normalised_objectives(ind)?;
            let obj_values = data.as_vec()?;
            // calculate the distances for all reference points
            let d_per = self
                .reference_points
                .iter()
                .map(|ref_point| {
                    perpendicular_distance(ref_point, obj_values).map_err(|e| {
                        OError::AlgorithmRun(
                            "NSGA3-AssociateToRefPoint".to_string(),
                            format!("Cannot calculate vector distance because: {}", e),
                        )
                    })
                })
                .collect::<Result<Vec<f64>, OError>>()?;

            // step 8 - get the reference point with the lowest minimum distance
            let (ri, min_d) = argmin(&d_per);
            ind.set_data(MIN_DISTANCE, DataValue::Real(min_d));
            ind.set_data(
                REF_POINT,
                DataValue::Vector(self.reference_points[ri].clone()),
            );
            ind.set_data(REF_POINT_INDEX, DataValue::USize(ri));
        }

        Ok(())
    }

    /// Check that the values in a vector are between 0 and 1 (i.e. all the values have been
    /// normalised).
    ///
    /// # Arguments
    ///
    /// * `points`: The point to check.
    /// * `what`: A string describing the vector of points.
    ///
    /// returns: `Result<(), OError>`
    fn check_bounds(points: &[f64], what: String) -> Result<(), OError> {
        if points.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(OError::AlgorithmRun(
                "NSGA3-AssociateToRefPoint".to_string(),
                format!(
                    "The values of the {} {:?} must be between 0 and 1",
                    what, points,
                ),
            ));
        }
        Ok(())
    }
}

/// This implements "Algorithm 4" in the paper which adds individuals from the last front to the new
/// population based on the reference point association and minimum distance.
struct Niching<'a> {
    /// The population being created at the current evolution. This is `$P_{t+1}$`
    new_population: &'a mut Population,
    /// Individuals from the last front $F_l$ to add to [`self.new_population`] based on reference
    /// point association and minimum distance.
    potential_individuals: &'a mut Vec<Individual>,
    /// The number of individuals to add to [`self.new_population`] to complete the evolution.
    missing_item_count: usize,
    /// The map mapping the reference point index to the number of individuals already associated in
    /// [`self.new_population`]
    rho_j: &'a mut HashMap<usize, usize>,
    /// The random number generator
    rng: &'a mut Box<dyn RngCore>,
}

impl<'a> Niching<'a> {
    fn new(
        new_population: &'a mut Population,
        potential_individuals: &'a mut Vec<Individual>,
        number_of_individuals_to_add: usize,
        rho_j: &'a mut HashMap<usize, usize>,
        rng: &'a mut Box<dyn RngCore>,
    ) -> Self {
        Self {
            new_population,
            potential_individuals,
            missing_item_count: number_of_individuals_to_add,
            rho_j,
            rng,
        }
    }

    /// Add new individuals to the population. This updates [`self.new_population`] by draining
    /// items from [`self.potential_individuals`]
    ///
    /// return: `Result<(), OError>`
    fn calculate(&mut self) -> Result<(), OError> {
        let mut k = 1;
        let mut excluded_ref_point_index: Vec<usize> = Vec::new();
        while k <= self.missing_item_count {
            // step 3 - select the reference point with the minimum rho_j counter
            let min_rho_j = *self
                .rho_j
                .iter()
                .min_by(|(_, v1), (_, v2)| v1.cmp(v2))
                .ok_or(OError::AlgorithmRun(
                    "NSGA3-Niching".to_string(),
                    "Empty rho_j set".to_string(),
                ))?
                .1;

            // collect all reference point indexes j with minimum rho_j - exclude reference points
            // that have no association with individuals in F_l (Z_r = Z_r/{j_hat}, step 15)
            let j_min_set: Vec<usize> = self
                .rho_j
                .iter()
                .filter_map(|(ref_index, ref_counter)| {
                    if *ref_counter == min_rho_j && !excluded_ref_point_index.contains(ref_index) {
                        Some(*ref_index)
                    } else {
                        None
                    }
                })
                .collect();

            // step 4
            let j_hat = if j_min_set.len() > 1 {
                // select point randomly
                *j_min_set.choose(self.rng.as_mut()).unwrap()
            } else {
                *j_min_set.first().unwrap()
            };

            // step 5 - individual in F_j linked to current reference point index j_hat
            let i_j: Vec<&Individual> = self
                .potential_individuals
                .iter()
                .filter(|ind| ind.get_data(REF_POINT_INDEX).unwrap() == DataValue::USize(j_hat))
                .collect();

            if !i_j.is_empty() {
                // here step 8 and 10 are combined. In step 8 ref point has no association with
                // points in P_{t+1}. In step 10, instead of randomly pick a point, the individual
                // closest to the reference point is used (as suggested in footnote 2).

                // step 8 or 10 - find individual in F_l with minimum distance
                let (new_ind_index, _) = argmin_by(&i_j, |(_, ind)| {
                    ind.get_data(MIN_DISTANCE).unwrap().as_real().unwrap()
                })
                .unwrap();

                // step 8 or 10 + 12 - Add individual with the shortest distance and remove
                // it from F_l
                self.new_population
                    .add_individual(self.potential_individuals.remove(new_ind_index));
                // step 12 - mark reference point as associated to new F_l's individual
                *self.rho_j.get_mut(&j_hat).unwrap() += 1;
                // step 13
                k += 1;
            } else {
                // step 15 - no point in F_l is associated with reference point indexed by j_hat.
                // j_hat will have no linked individual at this evolution. Skip it.
                debug!("Excluding ref point index {j_hat} - no candidates associated with it");
                excluded_ref_point_index.push(j_hat);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test_algorithms {
    use std::collections::HashMap;

    use float_cmp::assert_approx_eq;

    use crate::algorithms::nsga3::{
        AssociateToRefPoint, MIN_DISTANCE, Niching, Normalise, NORMALISED_OBJECTIVE_KEY, REF_POINT,
        REF_POINT_INDEX,
    };
    use crate::core::{DataValue, Individual, ObjectiveDirection, Population};
    use crate::core::test_utils::assert_approx_array_eq;
    use crate::core::utils::{get_rng, individuals_from_obj_values_dummy};
    use crate::utils::DasDarren1998;

    #[test]
    /// Test intercepts. Points were generated from numpy from uniform distribution with normal
    /// distributed noise on z coordinates (scale=1). Plane was generated to have slope of -2 in
    /// the x direction and -3 in the y direction.
    fn test_intercepts() {
        let points = vec![
            vec![3.3817863, 0.40604364, -2.2899773],
            vec![4.1741924, 0.92094903, -5.91434001],
            vec![3.42070899, 0.90266942, -3.81063094],
            vec![1.11301849, 0.94849208, 0.17140235],
            vec![9.08303894, 0.74599477, -16.14020622],
            vec![0.98976491, 0.84847939, 0.82864021],
            vec![7.53579489, 0.73723563, -11.72284018],
            vec![6.96274164, 0.59449793, -10.71963907],
            vec![5.60255823, 1.69973452, -12.49841699],
            vec![6.16815342, 0.66601692, -11.63169056],
        ];
        let intercepts = Normalise::calculate_plane_intercepts(&points)
            .unwrap()
            .unwrap();
        assert_approx_array_eq(&intercepts, &[3.38096778, 1.61009025, 7.58962871]);
    }

    #[test]
    /// Test AssociateToRefPoint that calculates the correct distances and reference point association
    fn test_association() {
        let das_darren = DasDarren1998::new(3, 4);
        let ref_points = das_darren.get_weights();

        let dummy_objectives = vec![[0.0, 0.0], [50.0, 50.0]];
        let mut individuals = individuals_from_obj_values_dummy(
            &dummy_objectives,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
        );
        // set normalised objectives
        individuals[0].set_data(
            NORMALISED_OBJECTIVE_KEY,
            DataValue::Vector(vec![0.95, 0.15, 0.15]),
        );
        individuals[1].set_data(
            NORMALISED_OBJECTIVE_KEY,
            DataValue::Vector(vec![0.1, 0.9, 0.1]),
        );

        let mut ass = AssociateToRefPoint::new(&mut individuals, &ref_points).unwrap();
        ass.calculate().unwrap();

        // 1st individual
        assert_approx_array_eq(
            individuals[0]
                .get_data(REF_POINT)
                .unwrap()
                .as_vec()
                .unwrap(),
            &[1.0, 0.0, 0.0],
        );
        assert_approx_eq!(
            f64,
            individuals[0]
                .get_data(MIN_DISTANCE)
                .unwrap()
                .as_real()
                .unwrap(),
            0.212132034355,
            epsilon = 0.0001
        );

        // 2nd individual
        assert_approx_array_eq(
            individuals[1]
                .get_data(REF_POINT)
                .unwrap()
                .as_vec()
                .unwrap(),
            &[0.0, 1.0, 0.0],
        );
        assert_approx_eq!(
            f64,
            individuals[1]
                .get_data(MIN_DISTANCE)
                .unwrap()
                .as_real()
                .unwrap(),
            0.1414213562,
            epsilon = 0.0001
        );
    }

    #[test]
    /// Check niching that adds point with min distance.
    fn test_niching() {
        env_logger::init();

        // create dummy population with 4 individuals
        let dummy_objectives = vec![[0.0, 0.0]; 2];
        let mut individuals = individuals_from_obj_values_dummy(
            &dummy_objectives,
            &[ObjectiveDirection::Minimise; 2],
        );
        let problem = individuals[0].problem().clone();
        let mut rho_j: HashMap<usize, usize> = HashMap::new();

        // link 2 individuals to 2 out of 4 reference points
        individuals[0].set_data(REF_POINT_INDEX, DataValue::USize(0));
        individuals[0].set_data(MIN_DISTANCE, DataValue::Real(0.1));
        rho_j.entry(0).or_insert(1);

        individuals[1].set_data(REF_POINT_INDEX, DataValue::USize(1));
        individuals[1].set_data(MIN_DISTANCE, DataValue::Real(0.2));
        rho_j.entry(1).or_insert(1);
        let mut pop = Population::new_with(individuals);

        // potential individuals - both are linked to ref_point #3 but ind_3 is closer
        let mut ind_3 = Individual::new(problem.clone());
        ind_3.set_data(REF_POINT_INDEX, DataValue::USize(2));
        ind_3.set_data(MIN_DISTANCE, DataValue::Real(0.4));

        let mut ind_4 = Individual::new(problem);
        ind_4.set_data(REF_POINT_INDEX, DataValue::USize(2));
        ind_4.set_data(MIN_DISTANCE, DataValue::Real(0.9));

        // counter is 0 for all other ref_points
        rho_j.entry(2).or_insert(0);
        rho_j.entry(3).or_insert(0);
        let mut potential_individuals = vec![ind_3, ind_4];
        let selected_ind = potential_individuals[0].clone();

        let mut rng = get_rng(Some(1));
        let mut n = Niching::new(
            &mut pop,
            &mut potential_individuals,
            1,
            &mut rho_j,
            &mut rng,
        );
        n.calculate().unwrap();

        // counter for ref_point #3 has increased
        assert_eq!(rho_j[&2_usize], 1_usize);
        // 3rd individual is added to the population
        assert_eq!(pop.len(), 3);
        // assert!(*pop.individual(2).unwrap() == selected_ind);
        assert_eq!(pop.individual(2).unwrap(), &selected_ind);
    }
}
