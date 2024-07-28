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
pub struct MaxGeneration(pub usize);

impl StoppingCondition<usize> for MaxGeneration {
    fn target(&self) -> usize {
        self.0
    }

    fn name() -> String {
        "maximum number of generations".to_string()
    }
}

/// Elapsed time after which a genetic algorithm terminates.
#[derive(Serialize, Deserialize, Clone)]
pub struct MaxDuration(pub Duration);

impl StoppingCondition<Duration> for MaxDuration {
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
    MaxDuration(MaxDuration),
    /// Set a maximum number of generations
    MaxGeneration(MaxGeneration),
}

impl StoppingConditionType {
    /// A name describing the stopping condition.
    pub fn name(&self) -> String {
        match self {
            StoppingConditionType::MaxDuration(_) => MaxDuration::name(),
            StoppingConditionType::MaxGeneration(_) => MaxGeneration::name(),
        }
    }
}
