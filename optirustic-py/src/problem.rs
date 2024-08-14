use std::collections::HashMap;

use pyo3::prelude::*;

use optirustic::core::{Constraint, Objective, VariableType};

use crate::constraint::PyConstraint;
use crate::objective::PyObjective;
use crate::variable::PyVariable;

#[pyclass(name = "Problem")]
#[derive(Debug, Clone)]
pub struct PyProblem {
    pub objectives: Vec<(String, Objective)>,
    pub constraints: Vec<(String, Constraint)>,
    pub variables: Vec<(String, VariableType)>,
    #[pyo3(get)]
    pub constraint_names: Vec<String>,
    #[pyo3(get)]
    pub variable_names: Vec<String>,
    #[pyo3(get)]
    pub objective_names: Vec<String>,
    #[pyo3(get)]
    pub number_of_objectives: usize,
    #[pyo3(get)]
    pub number_of_constraints: usize,
    #[pyo3(get)]
    pub number_of_variables: usize,
}

#[pymethods]
impl PyProblem {
    #[getter]
    pub fn variables(&self) -> PyResult<PyObject> {
        let mut dict = HashMap::new();
        for (name, v) in &self.variables {
            let var: PyVariable = v.into();
            dict.insert(name, var);
        }
        Ok(Python::with_gil(|py| dict.into_py(py)))
    }

    #[getter]
    pub fn objectives(&self) -> PyResult<PyObject> {
        let mut dict = HashMap::new();
        for (name, obj) in &self.objectives {
            let objective: PyObjective = obj.into();
            dict.insert(name, objective);
        }
        Ok(Python::with_gil(|py| dict.into_py(py)))
    }

    #[getter]
    pub fn constraints(&self) -> PyResult<PyObject> {
        let mut dict = HashMap::new();
        for (name, cons) in &self.constraints {
            let constraint: PyConstraint = cons.into();
            dict.insert(name, constraint);
        }
        Ok(Python::with_gil(|py| dict.into_py(py)))
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Problem(variables={}, objectives={}, constraints={})",
            self.number_of_variables, self.number_of_objectives, self.number_of_constraints
        ))
    }

    pub fn __str__(&self) -> String {
        self.__repr__().unwrap()
    }
}
