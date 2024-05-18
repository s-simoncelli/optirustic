pub use algorithm::Algorithm;
pub use nsga2::{NSGA2, NSGA2Arg};
pub use stopping_condition::{
    MaxDuration, MaxGeneration, StoppingCondition, StoppingConditionType,
};

pub mod algorithm;
pub mod nsga2;
pub mod stopping_condition;
