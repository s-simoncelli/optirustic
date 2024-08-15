use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use optirustic::algorithms::{
    Algorithm, AlgorithmExport, AlgorithmSerialisedExport, NSGA2Arg, NSGA3Arg, NSGA2 as RustNSGA2,
    NSGA3 as RustNSGA3,
};
use optirustic::core::OError;
use optirustic::metrics::HyperVolume;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::constraint::PyRelationalOperator;
use crate::individual::{PyData, PyIndividual};
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
            export_data: AlgorithmExport,
            #[pyo3(get)]
            problem: PyProblem,
            #[pyo3(get)]
            individuals: PyObject,
            #[pyo3(get)]
            took: PyObject,
            #[pyo3(get)]
            objectives: HashMap<String, Vec<f64>>,
            #[pyo3(get)]
            additional_data: Option<HashMap<String, PyData>>,
            #[pyo3(get)]
            exported_on: DateTime<Utc>,
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

                // Algorthm data
                let additional_data = file_data.additional_data.clone().map(|data_map| {
                    data_map
                        .iter()
                        .map(|(n, v)| {
                            let v: PyData = v.into();
                            (n.clone(), v)
                        })
                        .collect()
                });
                let exported_on = file_data.exported_on.clone();

                // Convert export
                let export_data: AlgorithmExport = file_data
                    .try_into()
                    .map_err(|e: OError| PyValueError::new_err(e.to_string()))?;

                // Problem
                let p = &export_data.problem;
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
                    kwargs.set_item("hours", export_data.took.hours)?;
                    kwargs.set_item("minutes", export_data.took.minutes)?;
                    kwargs.set_item("seconds", export_data.took.seconds)?;
                    let result = timedelta.call((), Some(&kwargs))?;
                    result.extract::<PyObject>()
                })?;

                // Individuals
                let mut list = vec![];
                for ind in &export_data.individuals {
                    let individual: PyIndividual = ind.into();
                    list.push(individual);
                }
                let individuals = Python::with_gil(|py| list.into_py(py));

                // All objective values by name
                let objectives = export_data
                    .get_objectives()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;

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
            pub fn generation(&self) -> usize {
                self.export_data.generation
            }

            #[getter]
            /// Get the algorithm name.
            pub fn algorithm(&self) -> String {
                self.export_data.algorithm.clone()
            }

            /// Calculate the hyper-volume metric.
            pub fn hyper_volume(&mut self, reference_point: Vec<f64>) -> PyResult<f64> {
                let hv = HyperVolume::from_individual(
                    &mut self.export_data.individuals,
                    &reference_point,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(hv)
            }

            /// Plot the Pareto front
            pub fn plot(&self) -> PyResult<PyObject> {
                let py_plot = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/utils/plot.py"));
                Python::with_gil(|py| {
                    let module = PyModule::from_code_bound(py, py_plot, "plot.py", "utils.plot")?;
                    let obj_count = self.problem.number_of_objectives;

                    let fun_name = if obj_count == 2 {
                        "plot_2d"
                    } else if obj_count == 3 {
                        "plot_3d"
                    } else {
                        "plot_parallel"
                    };
                    let fun: Py<PyAny> = module.getattr(fun_name)?.into();
                    fun.call1(
                        py,
                        (
                            self.objectives.clone(),
                            self.algorithm(),
                            self.generation(),
                            self.export_data.individuals.len(),
                        ),
                    )
                })
            }

            #[staticmethod]
            pub fn plot_convergence(
                folder: String,
                reference_point: Vec<f64>,
            ) -> PyResult<PyObject> {
                let folder = PathBuf::from(folder);
                let all_serialise_data = $type::read_json_files(&folder)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let data = HyperVolume::from_files(&all_serialise_data, &reference_point)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;

                let py_plot = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/utils/plot.py"));
                Python::with_gil(|py| {
                    let module = PyModule::from_code_bound(py, py_plot, "plot.py", "utils.plot")?;
                    let fun: Py<PyAny> = module.getattr("plot_convergence")?.into();
                    fun.call1(py, (data.generations(), data.values()))
                })
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
