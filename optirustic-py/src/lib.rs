use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use optirustic::algorithms::{
    Algorithm, AlgorithmExport, AlgorithmSerialisedExport, NSGA2Arg, NSGA3Arg, NSGA2 as RustNSGA2,
    NSGA3 as RustNSGA3,
};
use optirustic::core::OError;
use optirustic::metrics::HyperVolume;

use crate::constraint::PyRelationalOperator;
use crate::individual::PyIndividual;
use crate::objective::PyObjectiveDirection;
use crate::problem::PyProblem;

mod constraint;
mod individual;
mod objective;
mod problem;
mod variable;

/// Macro to generate python class for an algorithm data reader
macro_rules! create_interface {
    ($name: ident, $type: ident, $ArgType: ident) => {
        #[pyclass]
        pub struct $name {
            data: AlgorithmExport,
            #[pyo3(get)]
            problem: PyProblem,
            #[pyo3(get)]
            individuals: PyObject,
            #[pyo3(get)]
            took: PyObject,
            #[pyo3(get)]
            objectives: HashMap<String, Vec<f64>>,
        }

        #[pymethods]
        impl $name {
            #[new]
            /// Initialise the class
            pub fn new(file: String) -> PyResult<Self> {
                let path = PathBuf::from(file);
                let file_data: AlgorithmSerialisedExport<$ArgType> =
                    $type::read_json_file(&path)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;

                let data: AlgorithmExport = file_data
                    .try_into()
                    .map_err(|e: OError| PyValueError::new_err(e.to_string()))?;

                // Problem
                let p = &data.problem;
                let problem = PyProblem {
                    variables: p.variables(),
                    objectives: p.objectives(),
                    constraints: p.constraints(),
                    constraint_names: p.constraint_names(),
                    variable_names: p.variable_names(),
                    objective_names: p.objective_names(),
                    number_of_objectives: p.number_of_objectives(),
                    number_of_constraints: p.number_of_constraints(),
                    number_of_variables: p.number_of_variables(),
                };

                // Time taken
                let took = Python::with_gil(|py| {
                    let module = PyModule::import_bound(py, "datetime")?;

                    let timedelta = module.getattr("timedelta")?;
                    let kwargs = PyDict::new_bound(py);
                    kwargs.set_item("hours", data.took.hours)?;
                    kwargs.set_item("minutes", data.took.minutes)?;
                    kwargs.set_item("seconds", data.took.seconds)?;
                    let result = timedelta.call((), Some(&kwargs))?;
                    result.extract::<PyObject>()
                })?;

                // Individuals
                let mut list = vec![];
                for ind in &data.individuals {
                    let individual: PyIndividual = ind.into();
                    list.push(individual);
                }
                let individuals = Python::with_gil(|py| list.into_py(py));

                // All objective values by name
                let objectives = data
                    .get_objectives()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;

                Ok(Self {
                    data,
                    problem,
                    took,
                    individuals,
                    objectives,
                })
            }

            #[getter]
            /// Get the generation number.
            pub fn generation(&self) -> usize {
                self.data.generation
            }

            #[getter]
            /// Get the algorithm name.
            pub fn algorithm(&self) -> String {
                self.data.algorithm.clone()
            }

            /// Calculate the hyper-volume metric.
            pub fn hyper_volume(&mut self, reference_point: Vec<f64>) -> PyResult<f64> {
                let hv = HyperVolume::from_individual(&mut self.data.individuals, &reference_point)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(hv)
            }
        }
    };
}

// Register the python classes
create_interface!(NSGA2, RustNSGA2, NSGA2Arg);
create_interface!(NSGA3, RustNSGA3, NSGA3Arg);

#[pymodule(name = "optirustic")]
fn optirustic_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NSGA2>()?;
    m.add_class::<NSGA3>()?;
    m.add_class::<PyObjectiveDirection>()?;
    m.add_class::<PyRelationalOperator>()?;

    Ok(())
}
