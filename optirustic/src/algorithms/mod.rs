pub use algorithm::{Algorithm, AlgorithmExport, AlgorithmSerialisedExport, ExportHistory};
pub use nsga2::{NSGA2, NSGA2Arg};
pub use nsga3::{NSGA3, NSGA3Arg, Nsga3NumberOfIndividuals};
pub use stopping_condition::{
    MaxDuration, MaxGeneration, StoppingCondition, StoppingConditionType,
};

mod algorithm;
mod nsga2;
mod nsga3;
mod stopping_condition;
