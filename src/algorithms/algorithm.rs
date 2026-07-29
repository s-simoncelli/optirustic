use crate::operators::{PolynomialMutationArgs, SimulatedBinaryCrossoverArgs};
use chrono::{DateTime, Utc};
use log::{debug, info};
use rayon::{prelude::*, ThreadPool};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::fs::read_dir;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use std::{fmt, fs};

#[cfg(feature = "python")]
use crate::algorithms::{NSGA2Arg, NSGA3Arg};
use crate::utils::{elapsed, elapsed_as_string};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::algorithms::StoppingCondition;
use crate::core::{
    DataValue, Individual, IndividualExport, OError, ObjectiveDirection, Population, Problem,
    ProblemExport,
};

/// The struct to configure the number of threads. This can be set to the maximum available
/// threads, a given number or disabled.
#[derive(Serialize, Deserialize, Clone)]
pub enum NumThreads {
    /// Always use the maximum number of CPUs available. On systems with hyper-threading
    /// enabled this equals the number of logical cores and not the physical ones.
    Max,
    /// Set a maximum number.
    Use(usize),
    /// Disable parallel processing.
    Off,
}

impl Default for NumThreads {
    fn default() -> Self {
        NumThreads::Max
    }
}

/// The data with the elapsed time.
#[derive(Serialize, Deserialize, Debug)]
pub struct Elapsed {
    /// Elapsed hours.
    pub hours: u64,
    /// Elapsed minutes.
    pub minutes: u64,
    /// Elapsed seconds.
    pub seconds: u64,
}

/// The struct used to export an algorithm serialised data.
#[derive(Serialize, Deserialize, Debug)]
pub struct AlgorithmSerialisedExport<T: Serialize> {
    /// Specific options for an algorithm.
    pub options: T,
    /// The problem configuration.
    pub problem: ProblemExport,
    /// The individuals in the population.
    pub individuals: Vec<IndividualExport>,
    /// The generation the export was collected at.
    pub generation: u32,
    /// The number of function evaluations
    #[serde(default)]
    pub number_of_function_evaluations: u32,
    /// The algorithm name.
    pub algorithm: String,
    /// Any additional data exported by the algorithm.
    pub additional_data: Option<HashMap<String, DataValue>>,
    /// The time took to reach the `generation`.
    pub took: Elapsed,
    /// The date and time when the data was exported
    pub exported_on: DateTime<Utc>,
}

/// Implement a list of helper functions to get the problem and individuals.
impl<T: Serialize> AlgorithmSerialisedExport<T> {
    /// Build the [`Problem`] struct from serialised data. The problem will have a dummy
    /// [`Algorithm::evolve`] method.
    ///
    /// returns: `Result<Problem, OError>`
    pub fn problem(&self) -> Result<Problem, OError> {
        self.problem.clone().try_into()
    }

    /// Build the vector of [`Individual`] from serialised data. Each individual will have the
    /// objective, constraint, variable and data values from the serialised data.
    ///
    /// returns: `Result<Vec<Individual>, OError>`
    pub fn individuals(&self) -> Result<Vec<Individual>, OError> {
        let problem = Arc::new(self.problem()?);
        let mut individuals: Vec<Individual> = vec![];
        for individual_data in &self.individuals {
            let mut ind = Individual::new(problem.clone());
            for (name, value) in &individual_data.objective_values {
                ind.update_objective(name, *value)?;
            }
            for (name, value) in &individual_data.constraint_values {
                ind.update_constraint(name, *value)?;
            }
            for (name, value) in &individual_data.variable_values {
                ind.update_variable(name, value.clone())?;
            }
            for (name, value) in &individual_data.data {
                ind.set_data(name, value.clone());
            }
            ind.set_evaluated();
            individuals.push(ind);
        }

        Ok(individuals)
    }

    #[cfg(feature = "plotting")]
    /// Plot the Pareto front from the results.
    ///
    /// # Arguments
    ///
    /// * `destination`: The path and file name where to save the PNG file.
    /// * `fig_size`: An optional argument to change the figure size. When `None` this
    /// defaults to [800, 500] px.
    ///
    /// returns: `Result<f64, OError>`
    pub fn plot_front(
        &self,
        destination: &PathBuf,
        fig_size: Option<[u32; 2]>,
    ) -> Result<(), OError> {
        use gnuplot::AxesCommon;
        use gnuplot::{
            Figure,
            PlotOption::{Color, PointSymbol},
        };

        let fig_size = fig_size.unwrap_or([800, 500]);
        let title = format!(
            "Results for {} @ generation={} \nPopulation size={}",
            self.algorithm,
            self.generation,
            self.individuals.len()
        );
        let obj_count = self.problem.number_of_objectives;
        let obj_names = self.problem.objective_names.clone();
        let mut fg = Figure::new();
        let point_style = [PointSymbol('O'), Color("black".into())];

        if obj_count >= 2 && obj_count <= 3 {
            let obj1_name = obj_names.get(0).unwrap();
            let obj2_name = obj_names.get(1).unwrap();

            let obj1: Vec<f64> = self
                .individuals()?
                .iter()
                .map(|i| i.get_objective_value(obj1_name).unwrap())
                .collect();
            let obj2: Vec<f64> = self
                .individuals()?
                .iter()
                .map(|i| i.get_objective_value(obj2_name).unwrap())
                .collect();

            if obj_count == 2 {
                fg.axes2d()
                    .points(obj1, obj2, &point_style)
                    .set_x_label(obj1_name, &[])
                    .set_y_label(obj2_name, &[])
                    .set_x_grid(true)
                    .set_y_grid(true)
                    .set_title(&title, &[]);
            } else {
                let obj3_name = obj_names.get(2).unwrap();
                let obj3: Vec<f64> = self
                    .individuals()?
                    .iter()
                    .map(|i| i.get_objective_value(obj3_name).unwrap())
                    .collect();
                fg.axes3d()
                    .points(obj1, obj2, obj3, &point_style)
                    .set_x_label(obj1_name, &[])
                    .set_y_label(obj2_name, &[])
                    .set_z_label(obj3_name, &[])
                    .set_x_grid(true)
                    .set_y_grid(true)
                    .set_z_grid(true)
                    .set_view(40.0, 110.0)
                    .set_title(&title, &[]);
            }

            return fg
                .save_to_png(destination, fig_size[0], fig_size[1])
                .map_err(|e| {
                    OError::Generic(format!("Cannot save the chart because {}", e.to_string()))
                });
        } else {
            return Err(OError::Generic(
                "Plotting of Pareto front is only supported for a 2 or 3-objective problem"
                    .to_string(),
            ));
        }
    }
}

/// Convert the [`AlgorithmSerialisedExport`] to [`AlgorithmExport`]
impl<T: Serialize> TryInto<AlgorithmExport> for AlgorithmSerialisedExport<T> {
    type Error = OError;

    fn try_into(self) -> Result<AlgorithmExport, Self::Error> {
        let data = AlgorithmExport {
            problem: Arc::new(self.problem()?),
            individuals: self.individuals()?,
            generation: self.generation,
            algorithm: self.algorithm,
            number_of_function_evaluations: self.number_of_function_evaluations,
            took: self.took,
            additional_data: self.additional_data.unwrap_or_default(),
        };
        Ok(data)
    }
}

/// The enum to determine how to group vectorial results.
pub enum ExportVecGroupBy {
    /// Return a vector reporting the results for each individual. The length of each
    /// nested vector equals the number of objectives.
    Individual,
    /// Return a vector reporting the results for each objective. The length of each
    /// nested vector equals the number of individuals in the population.
    Objective,
}

/// The type of reference points used in the [`AdaptiveNSGA3`]
pub struct ReferencePointType {
    /// The location of the original reference points (i.e. the points that were used
    /// to initialise the algorithm). These are the points that are always preserved
    /// during the optimisation.
    pub original: Vec<Vec<f64>>,
    /// The new points that were added along with the original point to reduce crowding
    /// and to allocate one point for each Pareto-optimal solution.
    pub new: Vec<Vec<f64>>,
}

/// The struct used to export an algorithm data.
#[derive(Debug)]
pub struct AlgorithmExport {
    /// The problem.
    pub problem: Arc<Problem>,
    /// The individuals with the solutions, constraint and objective values at the current generation.
    pub individuals: Vec<Individual>,
    /// The generation number.
    pub generation: u32,
    /// The number of function evaluations
    pub number_of_function_evaluations: u32,
    /// The algorithm name used to evolve the individuals.
    pub algorithm: String,
    /// The time the algorithm took to reach the current generation.
    pub took: Elapsed,
    /// Additional data stored in the algorithm (such as reference points for [`crate::algorithms::NSGA3`]).
    pub additional_data: HashMap<String, DataValue>,
}

impl AlgorithmExport {
    /// Get the numbers stored in a real variable in all individuals. This returns an error if the
    /// variable does not exist or is not a real type.
    ///
    /// # Arguments
    ///
    /// * `name`: The variable name.
    ///
    /// returns: `Result<f64, OError>`
    pub fn get_real_variables(&self, name: &str) -> Result<Vec<f64>, OError> {
        self.individuals
            .iter()
            .map(|i| i.get_real_value(name))
            .collect()
    }

    /// Get the numbers stored in an objective for all individuals. This returns an error if the
    /// objective does not exist.
    ///
    /// # Arguments
    ///
    /// * `name`: The objective name.
    ///
    /// returns: `Result<f64, OError>`
    pub fn get_objective(&self, name: &str) -> Result<Vec<f64>, OError> {
        self.individuals
            .iter()
            .map(|i| i.get_objective_value(name))
            .collect()
    }

    /// Get the objective values grouped by objective name.
    ///
    /// returns: `Result<HashMap<String, Vec<f64>>, OError>`
    pub fn get_objectives(&self) -> Result<HashMap<String, Vec<f64>>, OError> {
        let mut map = HashMap::new();
        for name in self.problem.objective_names() {
            let data_vec = self
                .individuals
                .iter()
                .map(|i| i.get_objective_value(&name))
                .collect::<Result<Vec<f64>, OError>>()?;
            map.insert(name, data_vec);
        }
        Ok(map)
    }

    /// Reshape a vector containing data for each individuals [NxM] into a vector containing data
    /// for each objective [MxN].
    fn group_by_objective(&self, vector: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, OError> {
        let mut vector_by_objectives = vec![vec![]; self.problem.number_of_objectives()];
        for ind_objectives in vector.into_iter() {
            for obj_num in 0..self.problem.number_of_objectives() {
                let value = ind_objectives.get(obj_num);
                match value {
                    Some(v) => vector_by_objectives[obj_num].push(v.clone()),
                    None => {
                        return Err(OError::Generic(format!("Cannot group objective {obj_num}")));
                    }
                }
            }
        }
        Ok(vector_by_objectives)
    }

    /// Get the values of the normalised objective derived using the NSGA3 algorithm. This
    /// will return an error if the results were derived using another algorithm.
    ///
    /// # Arguments
    ///
    /// * `group_by`: Determine how the results are returned. When [`ExportVecGroupBy::Individual`],
    /// this returns a vector of vectors. The main vector have size equal to the number of
    /// individuals and the nested vectors have length equal to the objective number. When
    /// [`ExportVecGroupBy::Objective`], the vector is reshaped so that the main vector
    /// have size equal to the number of objectives and the nested vector length equal to
    /// the number of individuals.
    ///
    /// returns `Result<Vec<Vec<f64>>, OError>`: The vector with the value of the normalised
    /// objectives. The data is grouped based on the `group_by` value.
    pub fn get_nsga3_normalised_objectives(
        &self,
        group_by: ExportVecGroupBy,
    ) -> Result<Vec<Vec<f64>>, OError> {
        if self.algorithm != "NSGA3" {
            return Err(OError::Generic(
                "The exported data were not derived using NSGA3".to_string(),
            ));
        }

        let mut all_normalised_objectives_per_ind = Vec::new();
        for ind in self.individuals.iter() {
            all_normalised_objectives_per_ind
                .push(ind.get_data("normalised_objectives")?.as_f64_vec()?.clone());
        }

        let results = match group_by {
            ExportVecGroupBy::Individual => all_normalised_objectives_per_ind,
            //  self.group_by_objective(all_normalised_objectives_per_ind)?
            ExportVecGroupBy::Objective => {
                let mut all_normalised_objectives =
                    vec![vec![]; self.problem.number_of_objectives()];
                for ind_objectives in all_normalised_objectives_per_ind.into_iter() {
                    for obj_num in 0..self.problem.number_of_objectives() {
                        let value = ind_objectives.get(obj_num);
                        match value {
                            Some(v) => all_normalised_objectives[obj_num].push(v.clone()),
                            None => {
                                return Err(OError::Generic(format!(
                                    "Cannot group objective {obj_num}"
                                )));
                            }
                        }
                    }
                }
                all_normalised_objectives
            }
        };
        Ok(results)
    }

    /// Get the values of the reference points used by the NSGA3 algorithm. This
    /// will return an error if the results were derived using another algorithm.
    ///
    /// # Arguments
    ///
    /// * `group_by`: Determine how the results are returned. When [`ExportVecGroupBy::Individual`],
    /// this returns a vector of vectors. The main vector have size equal to the number of
    /// individuals and the nested vectors have length equal to the objective number. When
    /// [`ExportVecGroupBy::Objective`], the vector is reshaped so that the main vector
    /// have size equal to the number of objectives and the nested vector length equal to
    /// the number of individuals.
    ///
    /// returns `Result<Vec<Vec<f64>>, OError>`: The vector with the value of the reference
    /// points. The data is grouped based on the `group_by` value.
    pub fn get_nsga3_reference_points(
        &self,
        group_by: ExportVecGroupBy,
    ) -> Result<Vec<Vec<f64>>, OError> {
        if self.algorithm != "NSGA3" {
            return Err(OError::Generic(
                "The exported data were not derived using NSGA3".to_string(),
            ));
        }

        let ref_points_per_ind = match self.additional_data.get("reference_points") {
            Some(values) => {
                let mut ref_points = vec![];
                for point_vec in values.as_data_vec()?.into_iter() {
                    ref_points.push(point_vec.as_f64_vec()?.clone());
                }
                ref_points
            }
            None => {
                return Err(OError::Generic(format!("Cannot group reference_points")));
            }
        };

        let results: Vec<Vec<f64>> = match group_by {
            ExportVecGroupBy::Individual => ref_points_per_ind,
            ExportVecGroupBy::Objective => self.group_by_objective(ref_points_per_ind)?,
        };
        Ok(results)
    }

    /// Get the values of the reference points used by the adaptive NSGA3 algorithm. This
    /// function returns to group of points:
    /// - original point: these are the points that were used to initialise the algorithm
    /// and that are always preserved during the optimisation.
    /// - adaptive points: the new points that were added along with the original point
    /// to reduce crowding and to allocate one point for each Pareto-optimal solution.
    ///
    /// This will return an error if the results were derived using another algorithm.
    ///
    /// # Arguments
    ///
    /// * `group_by`: Determine how the results are returned. When [`ExportVecGroupBy::Individual`],
    /// this returns a vector of vectors. The main vector have size equal to the number of
    /// individuals and the nested vectors have length equal to the objective number. When
    /// [`ExportVecGroupBy::Objective`], the vector is reshaped so that the main vector
    /// have size equal to the number of objectives and the nested vector length equal to
    /// the number of individuals.
    ///
    /// returns `Result<Vec<Vec<f64>>, OError>`: The vector with the value of the reference
    /// points. The data is grouped based on the `group_by` value.
    pub fn get_adaptive_nsga3_reference_points(
        &self,
        group_by: ExportVecGroupBy,
    ) -> Result<ReferencePointType, OError> {
        let mut ref_points = self.get_nsga3_reference_points(ExportVecGroupBy::Individual)?;
        // split using the original number of points
        let original_point_count = match self.additional_data.get("number_of_reference_points") {
            Some(count) => count.as_real()?, // NOTE: this is deserialised as real
            None => {
                return Err(OError::Generic(
                    "Cannot find additional_data.number_of_reference_points".to_string(),
                ))
            }
        };
        let adaptive_points = ref_points.split_off(original_point_count as usize);

        let result = match group_by {
            ExportVecGroupBy::Individual => ReferencePointType {
                original: ref_points,
                new: adaptive_points,
            },
            ExportVecGroupBy::Objective => ReferencePointType {
                original: self.group_by_objective(ref_points)?,
                new: self.group_by_objective(adaptive_points)?,
            },
        };

        Ok(result)
    }
}

impl Display for AlgorithmExport {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{} at {} generations, took {} hours, {} minutes and {} seconds",
            self.algorithm, self.generation, self.took.hours, self.took.minutes, self.took.seconds
        )
    }
}

/// A struct with the options to configure the individual's history export. Export may be enabled in
/// an algorithm to save objectives, constraints and solutions to a file each time the generation
/// counter in [`Algorithm::generation`] increases by a certain step provided in `generation_step`.
/// Exporting history may be useful to track convergence and inspect an algorithm evolution.
#[cfg_attr(feature = "python", pyclass(get_all, from_py_object))]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExportHistory {
    /// Export the algorithm data each time the generation counter in [`Algorithm::generation`]
    /// increases by the provided step.
    generation_step: usize,
    /// Serialise the algorithm history and export the results to a JSON file in the given folder.
    destination: PathBuf,
}

impl ExportHistory {
    /// Initialise the export history configuration. This returns an error if the destination folder
    /// does not exist.
    ///
    /// # Arguments
    ///
    /// * `generation_step`: export the algorithm data each time the generation counter in a genetic
    //  algorithm increases by the provided step.
    /// * `destination`: serialise the algorithm history and export the results to a JSON file in
    ///    the given folder.
    ///
    /// returns: `Result<ExportHistory, OError>`
    pub fn new(generation_step: usize, destination: &PathBuf) -> Result<Self, OError> {
        if !destination.exists() {
            return Err(OError::Generic(format!(
                "The destination folder '{:?}' does not exist",
                destination
            )));
        }
        Ok(Self {
            generation_step,
            destination: destination.to_owned(),
        })
    }

    /// Get the destination where to save the files.
    ///
    /// returns: `&PathBuf`
    pub fn destination(&self) -> &PathBuf {
        &self.destination
    }

    /// Get the number of generation step the history file is saved.
    ///
    /// returns: `usize`
    pub fn generation_step(&self) -> usize {
        self.generation_step
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl ExportHistory {
    #[new]
    fn py_new(generation_step: usize, destination: PathBuf) -> PyResult<Self> {
        ExportHistory::new(generation_step, &destination)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// The trait to use to implement an algorithm.
pub trait Algorithm<AlgorithmOptions: Serialize + DeserializeOwned>: Display {
    /// Initialise the algorithm.
    ///
    /// return: `Result<(), OError>`
    fn initialise(&mut self) -> Result<(), OError>;

    /// Evolve the population.
    ///
    /// return: `Result<(), OError>`
    fn evolve(&mut self) -> Result<(), OError>;

    /// Return the current step of the algorithm evolution.
    ///
    /// return: `u32`.
    fn generation(&self) -> u32;

    /// Return the reference to the current step of the algorithm evolution.
    ///
    /// return: `&u32`.
    fn generation_as_ref(&self) -> &u32;

    /// Return the number of function evaluations. This is the number of times the algorithm evaluates
    /// an individual's objectives and constraints using [`Algorithm::evaluate_individual`]. If no
    /// new solutions/individuals are chosen by an algorithm, this counter will not increase, as past
    /// solutions are already evaluated.
    ///
    /// return: `u32`.
    fn number_of_function_evaluations(&self) -> u32;

    /// Return the algorithm name.
    ///
    /// return: `String`.
    fn name(&self) -> String;

    /// Get the time when the algorithm started.
    ///
    /// return: `&Instant`.
    fn start_time(&self) -> &Instant;

    /// Return the stopping condition.
    ///
    /// return: `&StoppingConditionType`.
    fn stopping_condition(&self) -> &StoppingCondition;

    /// Return the evolved population.
    ///
    /// return: `&Population`.
    fn population(&self) -> &Population;

    /// Return the problem.
    ///
    /// return: `Arc<Problem>`.
    fn problem(&self) -> Arc<Problem>;

    /// Return the history export configuration, if provided by the algorithm.
    ///
    /// return: `Option<&ExportHistory>`.
    fn export_history(&self) -> Option<&ExportHistory>;

    /// Export additional data stored by the algorithm.
    ///
    /// return: `Option<HashMap<String, DataValue>>`
    fn additional_export_data(&self) -> Option<HashMap<String, DataValue>> {
        None
    }

    /// Get the elapsed hours, minutes and seconds since the start of the algorithm.
    ///
    /// return: `[u64; 3]`. An array with the number of elapsed hours, minutes and seconds.
    fn elapsed(&self) -> [u64; 3] {
        elapsed(self.start_time().elapsed().as_secs())
    }

    /// Format the elapsed time as string.
    ///
    /// return: `String`.
    fn elapsed_as_string(&self) -> String {
        elapsed_as_string(self.start_time().elapsed().as_secs())
    }

    /// Get the ThreadPool instance.
    ///
    /// return: `&ThreadPool`.
    fn build_thread_pool(num_threads: NumThreads) -> Result<Option<ThreadPool>, OError> {
        let pool = match num_threads {
            NumThreads::Max | NumThreads::Use(_) => {
                let mut builder = rayon::ThreadPoolBuilder::new();
                if let NumThreads::Use(n) = num_threads {
                    builder = builder.num_threads(n);
                }
                let pool = builder.build().map_err(|e| {
                    OError::Generic(format!("cannot initialise the thread pool because {e}"))
                })?;
                Some(pool)
            }
            NumThreads::Off => None,
        };
        Ok(pool)
    }

    /// Evaluate the objectives and constraints for unevaluated individuals in the population. This
    /// updates the individual data only, runs the evaluation function in a plain loop and increase
    /// the `nfe` counter by the number of evaluated individuals.
    /// This returns an error if the evaluation function fails or the evaluation function does not
    /// provide a value for a problem constraints or objectives for one individual.
    /// Evaluation are performed in threads when `threads` is not [`NumThreads::None`].
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals to evaluate.
    /// * `nfe`: The reference to the number of function evaluation counter.
    /// * `thread_pool`: The `ThreadPool`.
    ///
    /// return `Result<usize, OError>`.
    fn do_evaluation(
        individuals: &mut [Individual],
        nfe: &mut u32,
        thread_pool: &Option<ThreadPool>,
    ) -> Result<(), OError> {
        let delta_nfe = Self::count_unevaluated(individuals);
        match thread_pool {
            Some(pool) => {
                pool.install(|| {
                    individuals
                        .into_par_iter()
                        .enumerate()
                        .try_for_each(|(idx, i)| Self::evaluate_individual(idx, i))
                })?;
            }
            None => individuals
                .iter_mut()
                .enumerate()
                .try_for_each(|(idx, i)| Self::evaluate_individual(idx, i))?,
        }

        *nfe += delta_nfe;
        Ok(())
    }

    /// Evaluate the objectives and constraints for one unevaluated individual. This returns an
    /// error if the evaluation function fails or the evaluation function does not provide a
    /// value for a problem constraints or objectives.
    ///
    /// # Arguments
    ///
    /// * `idx`: The individual index.
    /// * `individual`: The individual to evaluate.
    ///
    /// return `Result<(), OError>`
    fn evaluate_individual(idx: usize, i: &mut Individual) -> Result<(), OError> {
        debug!("Evaluating individual #{} - {:?}", idx + 1, i.variables());

        // skip evaluated solutions
        if i.is_evaluated() {
            debug!("Skipping evaluation for individual #{idx}. Already evaluated.");
            return Ok(());
        }
        let problem = i.problem();
        let results = problem
            .evaluator()
            .evaluate(i)
            .map_err(|e| OError::Evaluation(e.to_string()))?;

        // update the objectives and constraints for the individual
        debug!("Updating individual #{idx} objectives and constraints");
        for name in problem.objective_names() {
            if !results.objectives.contains_key(&name) {
                return Err(OError::Evaluation(format!(
                    "The evaluation function did non return the value for the objective named '{}'",
                    name
                )));
            };
            i.update_objective(&name, results.objectives[&name])?;
        }
        if let Some(constraints) = results.constraints {
            for name in problem.constraint_names() {
                if !constraints.contains_key(&name) {
                    return Err(OError::Evaluation(format!(
                        "The evaluation function did non return the value for the constraints named '{}'",
                        name
                    )));
                };

                i.update_constraint(&name, constraints[&name])?;
            }
        }
        i.set_evaluated();
        Ok(())
    }

    /// Count the number on unevaluated individuals.
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals to check.
    ///
    /// returns: `u32`
    fn count_unevaluated(individuals: &[Individual]) -> u32 {
        individuals
            .iter()
            .filter_map(|i| if !i.is_evaluated() { Some(1) } else { None })
            .sum()
    }
    /// Run the algorithm.
    ///
    /// return: `Result<(), OError>`
    fn run(&mut self) -> Result<(), OError> {
        info!("Starting {}", self.name());
        self.initialise()?;
        // Export at init
        if let Some(export) = self.export_history() {
            self.save_to_json(&export.destination, Some("Init"))?;
        }

        let mut history_gen_step: usize = 0;
        let mut avg_time = 0.0;
        'gen_loop: loop {
            // Export history
            if let Some(export) = self.export_history() {
                if history_gen_step == export.generation_step - 1 {
                    self.save_to_json(&export.destination, None)?;
                    history_gen_step = 0;
                } else {
                    history_gen_step += 1;
                }
            }

            // Evolve population
            info!("Generation #{}", self.generation());
            let now = SystemTime::now();
            self.evolve()?;
            if let Result::Ok(elapsed) = now.elapsed() {
                avg_time = (avg_time + elapsed.as_secs_f64()) / 2.0;
            }
            info!(
                "Evolved generation #{} - Elapsed Time: {}",
                self.generation(),
                self.elapsed_as_string()
            );

            // print time left. For vectorial stopping condition this cannot be calculated
            match self.stopping_condition() {
                StoppingCondition::MaxDurationAsMinutes(max_t) => {
                    let left = max_t * 60 - self.start_time().elapsed().as_secs() as u32;
                    info!("Approximate time left: {}", elapsed_as_string(left as u64));
                }
                StoppingCondition::MaxDurationAsHours(max_t) => {
                    let left = max_t * 60 * 24 - self.start_time().elapsed().as_secs() as u32;
                    info!("Approximate time left: {}", elapsed_as_string(left as u64));
                }
                StoppingCondition::MaxGeneration(gen) => {
                    let left = (gen - self.generation()) as f64 * avg_time;
                    info!("Approximate time left: {}", elapsed_as_string(left as u64));
                }
                StoppingCondition::MaxFunctionEvaluations(nfe) => {
                    let left = (nfe - self.number_of_function_evaluations()) as f64 * avg_time
                        / self.number_of_function_evaluations() as f64;
                    info!("Approximate time left: {}", elapsed_as_string(left as u64));
                }
                _ => {}
            }

            // Termination
            let cond = self.stopping_condition();
            let terminate = self.is_stopping_condition_met(cond)?;
            if terminate {
                // save last file
                if let Some(export) = self.export_history() {
                    self.save_to_json(&export.destination, Some("Final"))?;
                }

                info!("Stopping evolution because the {} was reached", cond.name());
                info!("Took {}", self.elapsed_as_string());
                break 'gen_loop;
            }

            info!("========================");
            debug!("");
            debug!("");
        }

        Ok(())
    }

    /// Check if the given stopping condition is met.
    ///
    /// # Arguments
    ///
    /// * `condition`: The stopping condition type.
    ///
    /// returns: `Result<bool, OError>`
    fn is_stopping_condition_met(&self, condition: &StoppingCondition) -> Result<bool, OError> {
        let is_met = match condition {
            StoppingCondition::MaxDurationAsMinutes(duration) => {
                *duration <= ((Instant::now().elapsed().as_secs() / 60) as u32)
            }
            StoppingCondition::MaxDurationAsHours(duration) => {
                *duration <= ((Instant::now().elapsed().as_secs() / 60 / 60) as u32)
            }
            StoppingCondition::MaxGeneration(generation) => *generation <= self.generation(),
            StoppingCondition::MaxFunctionEvaluations(nfe) => {
                *nfe <= self.number_of_function_evaluations()
            }
            StoppingCondition::Any(conditions) => {
                if StoppingCondition::has_nested_vector(conditions) {
                    return Err(OError::AlgorithmRun(
                        self.name(),
                        "A vector of stopping condition vector is not allowed".to_string(),
                    ));
                }
                conditions
                    .iter()
                    .any(|c| self.is_stopping_condition_met(c).unwrap())
            }
            StoppingCondition::All(conditions) => {
                if StoppingCondition::has_nested_vector(conditions) {
                    return Err(OError::AlgorithmRun(
                        self.name(),
                        "A vector of stopping condition vector is not allowed".to_string(),
                    ));
                }
                conditions
                    .iter()
                    .all(|c| self.is_stopping_condition_met(c).unwrap())
            }
            StoppingCondition::Function(custom_stopping_condition) => custom_stopping_condition
                .try_lock()
                .expect("Cannot get condition")
                .is_met(),
        };
        Ok(is_met)
    }

    /// Get the results of the run.
    ///
    /// return: `AlgorithmExport`.
    fn get_results(&self) -> AlgorithmExport {
        let [hours, minutes, seconds] = self.elapsed();
        AlgorithmExport {
            problem: self.problem(),
            individuals: self.population().individuals().to_vec(),
            generation: self.generation(),
            number_of_function_evaluations: self.number_of_function_evaluations(),
            algorithm: self.name(),
            took: Elapsed {
                hours,
                minutes,
                seconds,
            },
            additional_data: self.additional_export_data().unwrap_or_default(),
        }
    }

    fn algorithm_options(&self) -> AlgorithmOptions;

    /// Save the algorithm data (individuals' objective, variables and constraints, the problem,
    /// ...) to a JSON file. This returns an error if the file cannot be saved.
    ///
    /// # Arguments
    ///
    /// * `destination`: The path to the JSON file.
    /// * `file_prefix`: A prefix to prepend at the beginning of the file name. Empty when `None`.
    ///
    /// return `Result<(), OError>`
    fn save_to_json(&self, destination: &PathBuf, file_prefix: Option<&str>) -> Result<(), OError> {
        let file_prefix = file_prefix.unwrap_or("History");

        let [hours, minutes, seconds] = self.elapsed();
        let export = AlgorithmSerialisedExport {
            options: self.algorithm_options(),
            problem: self.problem().serialise(),
            individuals: self.population().serialise(),
            generation: self.generation(),
            number_of_function_evaluations: self.number_of_function_evaluations(),
            algorithm: self.name(),
            additional_data: self.additional_export_data(),
            took: Elapsed {
                hours,
                minutes,
                seconds,
            },
            exported_on: Utc::now(),
        };
        let data = serde_json::to_string_pretty(&export).map_err(|e| {
            OError::AlgorithmExport(format!(
                "The following error occurred while converting the history struct: {e}"
            ))
        })?;

        let mut file = destination.to_owned();

        file.push(format!(
            "{}_{}_gen{}.json",
            file_prefix,
            self.name(),
            self.generation()
        ));

        info!("Saving JSON file {:?}", file);
        fs::write(file, data).map_err(|e| {
            OError::AlgorithmExport(format!(
                "The following error occurred while exporting the history JSON file: {e}",
            ))
        })?;
        Ok(())
    }

    /// Read the results previously exported with [`Self::save_to_json`].
    ///
    /// # Arguments
    ///
    /// * `file`: The path to the JSON file exported from this library.
    ///
    /// returns: `Result<AlgorithmSerialisedExport<T>, OError>`
    fn read_json_file(
        file: &PathBuf,
    ) -> Result<AlgorithmSerialisedExport<AlgorithmOptions>, OError> {
        if !file.exists() {
            return Err(OError::File(
                file.to_path_buf(),
                "the file does not exist".to_string(),
            ));
        }
        let data = fs::File::open(file).map_err(|e| {
            OError::File(
                file.to_path_buf(),
                format!("cannot read the JSON file because: {e}"),
            )
        })?;

        let mut history: AlgorithmSerialisedExport<AlgorithmOptions> =
            serde_json::from_reader(data).map_err(|e| {
                OError::File(
                    file.to_path_buf(),
                    format!("cannot parse the JSON file because: {e}"),
                )
            })?;

        // invert sign of maximised objective values
        for ind in &mut history.individuals {
            for (name, value) in ind.objective_values.iter_mut() {
                if history.problem.objectives[name].direction() == ObjectiveDirection::Maximise {
                    *value *= -1.0;
                }
            }
        }

        Ok(history)
    }

    /// Read the results from files exported during an algorithm evolution. This returns an error if
    /// the path does not exist or does not contain valid JSON files.
    ///
    /// # Arguments
    ///
    /// * `folder`: The folder path to the JSON files.
    ///
    /// returns: `Result<Vec<AlgorithmSerialisedExport<T>>, OError>`
    fn read_json_files(
        folder: &PathBuf,
    ) -> Result<Vec<AlgorithmSerialisedExport<AlgorithmOptions>>, OError> {
        let json_files: Vec<_> = read_dir(folder)
            .map_err(|e| OError::Generic(format!("Cannot read folder because {e}")))?
            .filter_map(|res| res.ok())
            .map(|dir_entry| dir_entry.path())
            .filter_map(|path| {
                if path.extension().map_or(false, |ext| ext == "json")
                    & path.file_name().map_or(false, |name| {
                        name.to_str().unwrap().starts_with("History_")
                            | name.to_str().unwrap().starts_with("Final_")
                    })
                {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        let results = json_files
            .iter()
            .map(|file| Self::read_json_file(file))
            .collect::<Result<Vec<_>, OError>>()?;
        Ok(results)
    }

    /// Seed the population using the values of variables, objectives and constraints exported
    /// to a JSON file.
    ///
    /// # Arguments
    ///
    /// * `problem`: The problem.
    /// * `name`: The algorithm name.
    /// * `expected_individuals`: The number of individuals to expect in the file. If this does not
    ///     match the population size, being used in the algorithm, an error is thrown.
    /// * `file`: The path to the JSON file exported from this library.
    ///
    /// returns: `Result<Population, OError>`
    fn seed_population_from_file(
        problem: Arc<Problem>,
        name: &str,
        expected_individuals: usize,
        file: &PathBuf,
    ) -> Result<Population, OError> {
        let data = Self::read_json_file(file)?;

        // check number of variables
        if problem.number_of_variables() != data.problem.variables.len() {
            return Err(OError::AlgorithmInit(
                name.to_string(),
                format!(
                    "The number of variables from the history file ({}) does not \
                    match the number of variables ({}) defined in the problem",
                    data.problem.variables.len(),
                    problem.number_of_variables()
                ),
            ));
        }

        // check individuals
        if expected_individuals != data.individuals.len() {
            return Err(OError::AlgorithmInit(
                name.to_string(),
                format!(
                    "The number of individuals from the history file ({}) does not \
                    match the population size ({}) used in the algorithm",
                    data.problem.variables.len(),
                    problem.number_of_variables()
                ),
            ));
        }

        Population::deserialise(&data.individuals, problem.clone())
    }
}

/// Enum used to identify the chosen algorithm and its options in a python wrapper. Pass this
/// to another python class to then match and run an algorithm from Rust.
///
/// # Python example
/// ```python
/// # define the NSGA2 options
/// args = NSGA2Arg(
///     number_of_individuals=10,
///     stopping_condition=StoppingCondition.max_duration(3)
/// )
///
/// # initialise the enum
/// algo = Algorithm.nsga2(args)
/// ```
#[cfg(feature = "python")]
#[pyclass(from_py_object, name = "Algorithm")]
#[derive(Clone)]
#[allow(non_camel_case_types)]
pub enum PyAlgorithm {
    nsga2 { options: NSGA2Arg },
    nsga3 { options: NSGA3Arg },
    adaptive_nsga3 { options: NSGA3Arg },
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAlgorithm {
    fn __repr__(&self) -> PyResult<String> {
        let value = match self {
            PyAlgorithm::nsga2 { options } => {
                format!("NSGA2(options={:?})", options.__repr__()?)
            }
            PyAlgorithm::nsga3 { options } => {
                format!("NSGA3(options={:?})", options.__repr__()?)
            }
            PyAlgorithm::adaptive_nsga3 { options } => {
                format!("AdaptiveNSGA3(options={:?})", options.__repr__()?)
            }
        };
        Ok(value)
    }

    fn __str__(&self) -> String {
        self.__repr__().unwrap()
    }
}

// Macro to generate python class for an algorithm data reader
#[cfg(feature = "python")]
macro_rules! create_py_reader_interface {
    ($name: ident, $type: ident, $ArgType: ident) => {
        #[pyclass]
        pub struct $name {
            export_data: crate::algorithms::AlgorithmExport,
            #[pyo3(get)]
            problem: crate::core::PyProblem,
            #[pyo3(get)]
            individuals: Vec<Individual>,
            #[pyo3(get)]
            took: Py<PyAny>,
            #[pyo3(get)]
            objectives: std::collections::HashMap<String, Vec<f64>>,
            #[pyo3(get)]
            additional_data: Option<std::collections::HashMap<String, DataValue>>,
            #[pyo3(get)]
            exported_on: chrono::DateTime<chrono::Utc>,
        }

        #[pymethods]
        impl $name {
            #[new]
            /// Initialise the class
            pub fn new(file: PathBuf) -> PyResult<Self> {
                let path = PathBuf::from(file);
                let file_data: crate::algorithms::AlgorithmSerialisedExport<$ArgType> =
                    $type::read_json_file(&path)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

                // Algorthm data
                let additional_data = file_data.additional_data.clone();
                let exported_on = file_data.exported_on.clone();

                // Convert export
                let export_data: crate::algorithms::AlgorithmExport = file_data
                    .try_into()
                    .map_err(|e: OError| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

                // Problem
                let problem: crate::core::PyProblem = export_data.problem.as_ref().into();

                // Time taken
                let took = Python::attach(|py| -> PyResult<Py<PyAny>> {
                    let module = PyModule::import(py, "datetime")?;

                    let timedelta = module.getattr("timedelta")?;
                    let kwargs = pyo3::types::IntoPyDict::into_py_dict(
                        &[
                            ("hours", export_data.took.hours),
                            ("minutes", export_data.took.minutes),
                            ("seconds", export_data.took.seconds),
                        ],
                        py,
                    )?;
                    let result = timedelta.call((), Some(&kwargs))?;
                    Ok(result.extract::<Py<PyAny>>()?)
                })?;

                // Individuals
                let individuals = export_data.individuals.clone();

                // All objective values by name
                let objectives = export_data
                    .get_objectives()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

                Ok(Self {
                    export_data,
                    problem,
                    took,
                    individuals,
                    objectives,
                    additional_data,
                    exported_on,
                })
            }

            #[getter]
            /// Get the generation number.
            pub fn generation(&self) -> u32 {
                self.export_data.generation
            }

            #[getter]
            /// Get the algorithm name.
            pub fn algorithm(&self) -> String {
                self.export_data.algorithm.clone()
            }

            /// Calculate the hyper-volume metric.
            pub fn hyper_volume(&mut self, reference_point: Vec<f64>) -> PyResult<f64> {
                let hv = crate::metrics::HyperVolume::from_individuals(
                    &mut self.export_data.individuals,
                    &reference_point,
                )
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(hv)
            }

            /// Estimate the reference point from serialised data.
            #[pyo3(signature = (offset=None))]
            pub fn estimate_reference_point(&self, offset: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
                let individuals = &self.export_data.individuals;
                let ref_point =
                    crate::metrics::HyperVolume::estimate_reference_point(individuals, offset)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(ref_point)
            }

            /// Estimate the reference point from files.
            #[staticmethod]
            #[pyo3(signature = (folder, offset=None))]
            pub fn estimate_reference_point_from_files(
                folder: PathBuf,
                offset: Option<Vec<f64>>,
            ) -> PyResult<Vec<f64>> {
                let all_serialise_data_vec = $type::read_json_files(&folder)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let ref_point = crate::metrics::HyperVolume::estimate_reference_point_from_files(
                    &all_serialise_data_vec,
                    offset,
                )
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(ref_point)
            }

            #[staticmethod]
            pub fn convergence_data(
                folder: PathBuf,
                reference_point: Vec<f64>,
            ) -> PyResult<(Vec<u32>, Vec<chrono::DateTime<chrono::Utc>>, Vec<f64>)> {
                let all_serialise_data = $type::read_json_files(&folder)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let data =
                    crate::metrics::HyperVolume::from_files(&all_serialise_data, &reference_point)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

                Ok((data.generations(), data.times(), data.values()))
            }
        }
    };
}

// Export macro to parent module
#[cfg(feature = "python")]
pub(crate) use create_py_reader_interface;

/// Get a string listing the algorithm options.
///
/// # Arguments
///
/// * `problem`: The problem.
/// * `crossover_options`: The crossover operator options.
/// * `mutation_options`: The mutation operator options.
///
/// returns: `String`
pub fn algorithm_options_as_str(
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
            mutation_options.index_parameter, mutation_options.variable_probability,
        )
        .as_str(),
    );
    log_opts
}

// Custom Py object conversion
#[cfg(feature = "python")]
pub mod py {
    use crate::algorithms::NumThreads;
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use pyo3::types::PyString;

    #[pyclass(name = "NumThreads")]
    pub struct PyNumThreads;

    /// Convert a python object to `NumThreads`.
    impl FromPyObject<'_, '_> for NumThreads {
        type Error = PyErr;

        fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
            if obj.is_none() {
                Ok(NumThreads::default())
            } else if let Ok(x) = obj.extract::<usize>() {
                Ok(NumThreads::Use(x))
            } else if let Ok(x) = obj.extract::<String>() {
                if x == "Max" {
                    Ok(NumThreads::Max)
                } else if x == "Off" {
                    Ok(NumThreads::Off)
                } else {
                    Err(PyTypeError::new_err("Invalid string".to_string()))
                }
            } else {
                Err(PyTypeError::new_err("Invalid type".to_string()))
            }
        }
    }

    impl<'py> IntoPyObject<'py> for NumThreads {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr; // or Infallible if conversion cannot fail

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                NumThreads::Max => {
                    let s = PyString::new(py, "Max");
                    Ok(s.into_any())
                }
                NumThreads::Use(number) => {
                    let n = number.into_pyobject(py)?;
                    Ok(n.into_any())
                }
                NumThreads::Off => {
                    let s = PyString::new(py, "Off");
                    Ok(s.into_any())
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::env;
    use std::path::Path;
    use std::sync::Arc;

    use crate::algorithms::{Algorithm, NSGA2Arg, NumThreads, StoppingCondition, NSGA2};
    use crate::core::builtin_problems::{SCHProblem, ZTD1Problem};

    #[test]
    /// Test seed_population_from_file
    fn test_load_from_file() {
        let file = Path::new(&env::current_dir().unwrap())
            .join("examples")
            .join("results")
            .join("SCH_2obj_NSGA2_gen250.json");

        let problem = SCHProblem::create().unwrap();
        let pop = NSGA2::seed_population_from_file(Arc::new(problem), "NSGA2", 100, &file);
        assert!(pop.is_ok());
    }

    #[test]
    /// Test seed_population_from_file when the number of individuals is wrong.
    fn test_load_from_file_error() {
        let file = Path::new(&env::current_dir().unwrap())
            .join("examples")
            .join("results")
            .join("SCH_2obj_NSGA2_gen250.json");

        let problem = SCHProblem::create().unwrap();
        let pop = NSGA2::seed_population_from_file(Arc::new(problem), "NSGA2", 10, &file);
        assert!(pop
            .err()
            .unwrap()
            .to_string()
            .contains("number of individuals from the history file"));
    }

    #[test]
    /// Test seed_population_from_file when the wrong problem is used.
    fn test_load_from_file_wrong_problem() {
        let file = Path::new(&env::current_dir().unwrap())
            .join("examples")
            .join("results")
            .join("SCH_2obj_NSGA2_gen250.json");

        let problem = ZTD1Problem::create(30).unwrap();
        let pop = NSGA2::seed_population_from_file(Arc::new(problem), "NSGA2", 10, &file);

        assert!(pop
            .err()
            .unwrap()
            .to_string()
            .contains("number of variables from the history file"));
    }

    #[test]
    /// Test StoppingConditionType::MaxGeneration
    fn test_stopping_condition_max_generation() {
        let problem = SCHProblem::create().unwrap();
        let args = NSGA2Arg {
            number_of_individuals: 10,
            stopping_condition: StoppingCondition::MaxGeneration(20),
            crossover_operator_options: None,
            mutation_operator_options: None,
            threads: NumThreads::Off,
            export_history: None,
            resume_from_file: None,
            seed: Some(10),
        };
        let mut algo = NSGA2::new(Arc::new(problem), args).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        assert_eq!(results.generation, 20);
    }

    #[test]
    /// Test StoppingConditionType::MaxFunctionEvaluations
    fn test_stopping_condition_max_nfe() {
        let problem = SCHProblem::create().unwrap();
        let args = NSGA2Arg {
            number_of_individuals: 10,
            stopping_condition: StoppingCondition::MaxFunctionEvaluations(20),
            crossover_operator_options: None,
            mutation_operator_options: None,
            threads: NumThreads::Off,
            export_history: None,
            resume_from_file: None,
            seed: Some(10),
        };
        let mut algo = NSGA2::new(Arc::new(problem), args).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        assert_eq!(results.number_of_function_evaluations, 20);
        assert_eq!(results.generation, 2);
    }

    #[test]
    /// Test StoppingConditionType::Any
    fn test_stopping_condition_any() {
        let problem = SCHProblem::create().unwrap();
        let args = NSGA2Arg {
            number_of_individuals: 10,
            stopping_condition: StoppingCondition::Any(vec![
                StoppingCondition::MaxFunctionEvaluations(20),
                StoppingCondition::MaxGeneration(10),
            ]),
            crossover_operator_options: None,
            mutation_operator_options: None,
            threads: NumThreads::Off,
            export_history: None,
            resume_from_file: None,
            seed: Some(10),
        };
        let mut algo = NSGA2::new(Arc::new(problem), args).unwrap();
        algo.run().unwrap();
        let results = algo.get_results();

        assert_eq!(results.number_of_function_evaluations, 20);
        assert_eq!(results.generation, 2);
    }
}

#[cfg(feature = "python")]
#[cfg(test)]
mod test_python_api {
    use std::env;
    use std::error::Error;
    use std::path::Path;

    use crate::algorithms::NSGA3Data;
    use crate::utils::{DasDarren1998, NumberOfPartitions, TwoLayerPartitions};
    use float_cmp::assert_approx_eq;
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    #[test]
    /// test the NSGA*Data class.
    fn test_reader() -> Result<(), Box<dyn Error>> {
        Python::attach(|py| -> Result<(), Box<dyn Error>> {
            let file = Path::new(&env::current_dir()?)
                .join("examples")
                .join("results")
                .join("DTLZ1_3obj_NSGA3_gen400.json");
            let reader = Py::new(py, NSGA3Data::new(file)?)?;

            // check problem
            let problem = reader.getattr(py, "problem")?;
            assert_eq!(
                problem
                    .getattr(py, "number_of_variables")?
                    .extract::<i32>(py)?,
                7
            );
            assert_eq!(
                problem
                    .getattr(py, "variables")?
                    .call_method1(py, "get", ("x1".to_string(),))?
                    .getattr(py, "min_value")?
                    .extract::<f64>(py)?,
                0.0
            );
            assert_eq!(
                problem
                    .getattr(py, "objectives")?
                    .call_method0(py, "__len__")?
                    .extract::<i32>(py)?,
                3
            );

            // check reader props
            assert_eq!(
                reader.getattr(py, "algorithm")?.extract::<String>(py)?,
                "NSGA3".to_string()
            );
            assert_eq!(reader.getattr(py, "generation")?.extract::<i32>(py)?, 400);
            assert_eq!(
                reader
                    .getattr(py, "exported_on")?
                    .getattr(py, "day")?
                    .extract::<u32>(py)?,
                10
            );

            // test individuals
            let ind = reader
                .getattr(py, "individuals")?
                .call_method1(py, "pop", (0,))?;
            assert_eq!(
                ind.call_method0(py, "constraint_violation")?
                    .extract::<f32>(py)?,
                0.0
            );
            assert_approx_eq!(
                f32,
                ind.call_method1(py, "get_objective_value", ("f2",))?
                    .extract::<f32>(py)?,
                0.167,
                epsilon = 0.001
            );
            assert!(ind
                .call_method0(py, "variables")?
                .call_method0(py, "keys")?
                .call_method1(py, "__contains__", ("x5".to_string(),))?
                .extract::<bool>(py)?);
            assert!(ind
                .call_method0(py, "data")?
                .call_method0(py, "keys")?
                .call_method1(py, "__contains__", ("reference_point_index".to_string(),))?
                .extract::<bool>(py)?);
            assert_approx_eq!(
                f32,
                reader
                    .call_method1(py, "hyper_volume", (PyList::new(py, [100, 100, 100])?,),)?
                    .extract::<f32>(py)?,
                999999.97,
                epsilon = 0.001
            );
            Ok(())
        })
    }

    #[test]
    /// Test the DasDarren1998 class
    fn test_hypervolume() -> Result<(), Box<dyn Error>> {
        Python::attach(|py| -> Result<(), Box<dyn Error>> {
            let hv = Py::new(py, DasDarren1998::new(3, &NumberOfPartitions::OneLayer(5))?)?;
            assert_eq!(
                hv.call_method0(py, "calculate")?
                    .call_method0(py, "__len__")?
                    .extract::<u32>(py)?,
                21
            );

            // assert len(ds.calculate()) == 25
            let hv = Py::new(
                py,
                DasDarren1998::new(
                    3,
                    &&NumberOfPartitions::TwoLayers(TwoLayerPartitions {
                        boundary_layer: 3,
                        inner_layer: 4,
                        scaling: None,
                    }),
                )?,
            )?;
            assert_eq!(
                hv.call_method0(py, "calculate")?
                    .call_method0(py, "__len__")?
                    .extract::<u32>(py)?,
                25
            );

            Ok(())
        })
    }
}
