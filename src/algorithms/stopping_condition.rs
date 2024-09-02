use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Trait to define a condition that causes an algorithm to terminate.
pub trait StoppingCondition<T: PartialOrd> {
    /// The target value of the stopping condition.
    fn target(&self) -> T;

    /// Whether the stopping condition is met.
    fn is_met(&self, current: T) -> bool {
        self.target() <= current
    }

    /// A name describing the stopping condition.
    fn name() -> String;
}

/// Number of generations after which a genetic algorithm terminates.
#[derive(Serialize, Deserialize, Clone)]
pub struct MaxGenerationValue(pub usize);

impl StoppingCondition<usize> for MaxGenerationValue {
    fn target(&self) -> usize {
        self.0
    }

    fn name() -> String {
        "maximum number of generations".to_string()
    }
}

/// Number of function evaluations after which a genetic algorithm terminates.
#[derive(Serialize, Deserialize, Clone)]
pub struct MaxFunctionEvaluationValue(pub usize);

impl StoppingCondition<usize> for MaxFunctionEvaluationValue {
    fn target(&self) -> usize {
        self.0
    }

    fn name() -> String {
        "maximum number of function evaluations".to_string()
    }
}

/// Elapsed time after which a genetic algorithm terminates.
#[derive(Serialize, Deserialize, Clone)]
pub struct MaxDurationValue(pub Duration);

impl StoppingCondition<Duration> for MaxDurationValue {
    fn target(&self) -> Duration {
        self.0
    }

    fn name() -> String {
        "maximum duration".to_string()
    }
}

/// The type of stopping condition. Pick one type to inform the algorithm how/when it should
/// terminate the population evolution.
#[derive(Serialize, Deserialize, Clone)]
pub enum StoppingConditionType {
    /// Set a maximum duration
    MaxDuration(MaxDurationValue),
    /// Set a maximum number of generations
    MaxGeneration(MaxGenerationValue),
    /// Set a maximum number of function evaluations
    MaxFunctionEvaluations(MaxFunctionEvaluationValue),
    /// Stop when at least on condition is met
    Any(Vec<StoppingConditionType>),
    /// Stop when all on conditions are met
    All(Vec<StoppingConditionType>),
}

impl StoppingConditionType {
    /// A name describing the stopping condition.
    ///
    /// returns: `String`
    pub fn name(&self) -> String {
        match self {
            StoppingConditionType::MaxDuration(_) => MaxDurationValue::name(),
            StoppingConditionType::MaxGeneration(_) => MaxGenerationValue::name(),
            StoppingConditionType::MaxFunctionEvaluations(_) => MaxFunctionEvaluationValue::name(),
            StoppingConditionType::Any(s) => s
                .iter()
                .map(|cond| cond.name())
                .collect::<Vec<String>>()
                .join(" OR "),
            StoppingConditionType::All(s) => s
                .iter()
                .map(|cond| cond.name())
                .collect::<Vec<String>>()
                .join(" AND "),
        }
    }

    /// Check whether the stopping condition is a vector and has nested vector in it.
    ///
    /// # Arguments
    ///
    /// * `conditions`: A vector of stopping conditions.
    ///
    /// returns: `bool`
    pub fn has_nested_vector(conditions: &[StoppingConditionType]) -> bool {
        conditions.iter().any(|c| match c {
            StoppingConditionType::Any(_) | StoppingConditionType::All(_) => true,
            _ => false,
        })
    }
}
