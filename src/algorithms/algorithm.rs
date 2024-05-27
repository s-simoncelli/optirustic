use std::{fmt, fs};
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use log::{debug, info};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::algorithms::{StoppingCondition, StoppingConditionType};
use crate::core::{Individual, IndividualExport, Population, Problem, ProblemExport};
use crate::core::error::OError;

#[derive(Serialize, Deserialize, Debug)]
/// The data with the elapsed time.
pub struct Elapsed {
    /// Elapsed hours.
    hours: u64,
    /// Elapsed minutes.
    minutes: u64,
    /// Elapsed seconds.
    seconds: u64,
}

#[derive(Serialize, Deserialize, Debug)]
/// The struct used to export the algorithm data to JSON file.
pub struct AlgorithmSerialisedExport<T: Serialize> {
    pub options: T,
    pub problem: ProblemExport,
    pub individuals: Vec<IndividualExport>,
    pub generation: usize,
    pub algorithm: String,
    pub took: Elapsed,
}

/// The struct used to export the algorithm data.
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
}

/// A struct with the options to configure the individual's history export. Export may be enabled in
/// an algorithm to save objectives, constraints and solutions to a file each time the generation
/// counter in [`Algorithm::generation`] increases by a certain step provided in `generation_step`.
/// Exporting history may be useful to track convergence and inspect an algorithm evolution.
#[derive(Serialize, Clone)]
pub struct ExportHistory {
    /// Export the algorithm data each time the generation counter in [`Algorithm::generation`]
    /// increases by the provided step.
    generation_step: usize,
    /// Serialise the algorithm history and export the results to a JSON file in the given folder.
    destination: PathBuf,
}

impl ExportHistory {
    /// Initialise the export history configuration. This returns an error if the destination folder
    /// does not exists.
    ///
    /// # Arguments
    ///
    /// * `generation_step`: export the algorithm data each time the generation counter in a genetic
    //  algorithm increases by the provided step.
    /// * `destination`: serialise the algorithm history and export the results to a JSON file in
    /// the given folder.
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
pub trait Algorithm<AlgorithmOptions: Serialize>: Display {
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

        let mut history_gen_step: usize = 0;
        loop {
            // Evolve population
            info!("Generation #{}", self.generation());
            self.evolve()?;
            info!(
                "Evolved generation #{} - Elapsed Time: {}",
                self.generation(),
                self.elapsed_as_string()
            );

            // Export history
            if let Some(export) = self.export_history() {
                if history_gen_step >= export.generation_step {
                    self.save_to_json(&export.destination)?;
                    history_gen_step = 0;
                } else {
                    history_gen_step += 1;
                }
            }

            // Termination
            let cond = self.stopping_condition();
            let terminate = match &cond {
                StoppingConditionType::MaxDuration(t) => t.is_met(Instant::now().elapsed()),
                StoppingConditionType::MaxGeneration(t) => t.is_met(self.generation()),
            };
            if terminate {
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
            individuals: self.population().0.clone(),
            generation: self.generation(),
            algorithm: self.name(),
            took: Elapsed {
                hours,
                minutes,
                seconds,
            },
        }
    }

    fn algorithm_options(&self) -> &AlgorithmOptions;

    /// Save the algorithm data (individuals' objective, variables and constraints, the problem,
    /// ...) to a JSON file.
    ///
    /// # Arguments
    ///
    /// * `destination`: The path to the JSON file.
    ///
    /// return `Result<(), OError>`
    fn save_to_json(&self, destination: &PathBuf) -> Result<(), OError> {
        let [hours, minutes, seconds] = self.elapsed();
        let export = AlgorithmSerialisedExport {
            options: self.algorithm_options(),
            problem: self.problem().serialise(),
            individuals: self.population().serialise(),
            generation: self.generation(),
            algorithm: self.name(),
            took: Elapsed {
                hours,
                minutes,
                seconds,
            },
        };
        let data = serde_json::to_string_pretty(&export)
            .map_err(|e| OError::AlgorithmExport(e.to_string()))?;

        fs::write(destination, data).map_err(|e| OError::AlgorithmExport(e.to_string()))?;
        Ok(())
    }
}
