use std::fs;
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
    hours: u64,
    minutes: u64,
    seconds: u64,
}

#[derive(Serialize, Deserialize, Debug)]
/// The struct used to export the algorithm data to JSON file.
pub struct AlgorithmSerialisedExport {
    pub problem: ProblemExport,
    pub individuals: Vec<IndividualExport>,
    pub generation: usize,
    pub algorithm: String,
    pub took: Elapsed,
}

#[derive(Debug)]
/// The struct used to export the algorithm data.
pub struct AlgorithmExport {
    pub problem: Arc<Problem>,
    pub individuals: Vec<Individual>,
    pub generation: usize,
    pub algorithm: String,
    pub took: Elapsed,
}

/// The trait to use to implement an algorithm.
pub trait Algorithm {
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
    /// return: `Instant`.
    fn start_time(&self) -> Instant;

    /// Return the stopping condition.
    ///
    /// return: `StoppingConditionType`.
    fn stopping_condition(&self) -> StoppingConditionType;

    /// Return the evolved population.
    ///
    /// return: `Population`.
    fn population(&self) -> Population;

    /// Return the problem.
    ///
    /// return: `Arc<Problem>`.
    fn problem(&self) -> Arc<Problem>;

    /// Get the elapsed hours, minutes and seconds since the start of the algorithm.
    ///
    /// return: `[u64; 3]`.
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
            i.update_objective(&name, results.objectives[&name])?;
        }
        if let Some(constraints) = results.constraints {
            for name in problem.constraint_names() {
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

        loop {
            info!("Generation #{}", self.generation());
            self.evolve()?;
            info!(
                "Evolved generation #{} - Elapsed Time: {}",
                self.generation(),
                self.elapsed_as_string()
            );

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
            individuals: self.population().0,
            generation: self.generation(),
            algorithm: self.name(),
            took: Elapsed {
                hours,
                minutes,
                seconds,
            },
        }
    }

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
