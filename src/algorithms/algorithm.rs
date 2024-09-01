use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::fs::read_dir;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use std::{fmt, fs};

use chrono::{DateTime, Utc};
use log::{debug, info};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::algorithms::{StoppingCondition, StoppingConditionType};
use crate::core::{
    DataValue, Individual, IndividualExport, OError, ObjectiveDirection, Population, Problem,
    ProblemExport,
};

#[derive(Serialize, Deserialize, Debug)]
/// The data with the elapsed time.
pub struct Elapsed {
    /// Elapsed hours.
    pub hours: u64,
    /// Elapsed minutes.
    pub minutes: u64,
    /// Elapsed seconds.
    pub seconds: u64,
}

#[derive(Serialize, Deserialize, Debug)]
/// The struct used to export an algorithm serialised data.
pub struct AlgorithmSerialisedExport<T: Serialize> {
    /// Specific options for an algorithm.
    pub options: T,
    /// The problem configuration.
    pub problem: ProblemExport,
    /// The individuals in the population.
    pub individuals: Vec<IndividualExport>,
    /// The generation the export was collected at.
    pub generation: usize,
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
            took: self.took,
            additional_data: self.additional_data.unwrap_or_default(),
        };
        Ok(data)
    }
}

/// The struct used to export an algorithm data.
#[derive(Debug)]
pub struct AlgorithmExport {
    /// The problem.
    pub problem: Arc<Problem>,
    /// The individuals with the solutions, constraint and objective values at the current generation.
    pub individuals: Vec<Individual>,
    /// The generation number.
    pub generation: usize,
    /// The algorithm name used to evolve the individuals.
    pub algorithm: String,
    /// The time the algorithm took to reach the current generation.
    pub took: Elapsed,
    /// Additional data stored in the algorithm (such as reference points for [`NSGA3`]).
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
}

/// A struct with the options to configure the individual's history export. Export may be enabled in
/// an algorithm to save objectives, constraints and solutions to a file each time the generation
/// counter in [`Algorithm::generation`] increases by a certain step provided in `generation_step`.
/// Exporting history may be useful to track convergence and inspect an algorithm evolution.
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
    /// return: `usize`.
    fn generation(&self) -> usize;

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
    fn stopping_condition(&self) -> &StoppingConditionType;

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
        let duration = self.start_time().elapsed();
        let seconds = duration.as_secs() % 60;
        let minutes = (duration.as_secs() / 60) % 60;
        let hours = (duration.as_secs() / 60) / 60;
        [hours, minutes, seconds]
    }

    /// Format the elapsed time as string.
    ///
    /// return: `String`.
    fn elapsed_as_string(&self) -> String {
        let [hours, minutes, seconds] = self.elapsed();
        format!(
            "{:0>2} hours, {:0>2} minutes and {:0>2} seconds",
            hours, minutes, seconds
        )
    }

    /// Evaluate the objectives and constraints for unevaluated individuals in the population. This
    /// updates the individual data only and runs the evaluation function in threads. This returns
    /// an error if the evaluation function fails or the evaluation function does not provide a
    /// value for a problem constraints or objectives for one individual.
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals to evaluate.
    ///
    /// return `Result<(), OError>`
    fn do_parallel_evaluation(individuals: &mut [Individual]) -> Result<(), OError> {
        individuals
            .into_par_iter()
            .enumerate()
            .try_for_each(|(idx, i)| Self::evaluate_individual(idx, i))?;
        Ok(())
    }

    /// Evaluate the objectives and constraints for unevaluated individuals in the population. This
    /// updates the individual data only and runs the evaluation function in a plain loop. This
    /// returns an error if the evaluation function fails or the evaluation function does not
    /// provide a value for a problem constraints or objectives for one individual.
    /// Evaluation may be performed in threads using [`Self::do_parallel_evaluation`].
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals to evaluate.
    ///
    /// return `Result<(), OError>`
    fn do_evaluation(individuals: &mut [Individual]) -> Result<(), OError> {
        individuals
            .iter_mut()
            .enumerate()
            .try_for_each(|(idx, i)| Self::evaluate_individual(idx, i))?;
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
        loop {
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
            self.evolve()?;
            info!(
                "Evolved generation #{} - Elapsed Time: {}",
                self.generation(),
                self.elapsed_as_string()
            );
            debug!("========================");
            debug!("");
            debug!("");

            // Termination
            let cond = self.stopping_condition();
            let terminate = match &cond {
                StoppingConditionType::MaxDuration(t) => t.is_met(Instant::now().elapsed()),
                StoppingConditionType::MaxGeneration(t) => t.is_met(self.generation()),
            };
            if terminate {
                // save last file
                if let Some(export) = self.export_history() {
                    self.save_to_json(&export.destination, Some("Final"))?;
                }

                info!("Stopping evolution because the {} was reached", cond.name());
                info!("Took {}", self.elapsed_as_string());
                break;
            }
        }

        Ok(())
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
                if path.extension().map_or(false, |ext| ext == "json") {
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

#[cfg(test)]
mod test {
    use std::env;
    use std::path::Path;
    use std::sync::Arc;

    use crate::algorithms::{Algorithm, NSGA2};
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
}
