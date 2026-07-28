pub use a_nsga3::AdaptiveNSGA3;
pub use algorithm::{
    Algorithm, AlgorithmExport, AlgorithmSerialisedExport, ExportHistory, ExportVecGroupBy,
    NumThreads,
};
pub use esp_nsga2::EspNSGA2Arg;
pub use nsga2::{NSGA2Arg, CROWDING_DIST_KEY, NSGA2};
pub use nsga3::{NSGA3Arg, Nsga3NumberOfIndividuals, NSGA3};
pub use stopping_condition::StoppingCondition;

#[cfg(feature = "python")]
use algorithm::create_py_reader_interface;
#[cfg(feature = "python")]
pub use algorithm::py::PyNumThreads;
#[cfg(feature = "python")]
pub use algorithm::PyAlgorithm;
#[cfg(feature = "python")]
pub use nsga2::NSGA2Data;
#[cfg(feature = "python")]
pub use nsga3::NSGA3Data;
#[cfg(feature = "python")]
pub use stopping_condition::py::{PyStoppingConditionMap, PyStoppingConditionValue};

mod a_nsga3;
mod algorithm;
mod esp_nsga2;
mod nsga2;
mod nsga3;
mod stopping_condition;
