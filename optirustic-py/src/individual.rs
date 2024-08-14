use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use optirustic::core::{DataValue, Individual, VariableValue};

/// Convert `VariableValue` to `v`. Note this cannot be done with the `IntoPy` trait because
/// this crate does not own `VariableValue`.
fn variable_value_to_py(value: &VariableValue, py: Python<'_>) -> PyObject {
    match value {
        VariableValue::Real(v) => v.into_py(py),
        VariableValue::Integer(v) => v.into_py(py),
        VariableValue::Choice(v) => v.into_py(py),
        VariableValue::Boolean(v) => v.into_py(py),
    }
}

/// Struct holding data
#[derive(Clone)]
enum PyData {
    Real(f64),
    Integer(i64),
    USize(usize),
    Vector(Vec<f64>),
    DataVector(Vec<PyData>),
    Map(HashMap<String, PyData>),
}

/// Convert from `PyData` to `PyObject`
impl IntoPy<PyObject> for PyData {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match &self {
            PyData::Real(v) => v.into_py(py),
            PyData::Integer(v) => v.into_py(py),
            PyData::USize(v) => v.into_py(py),
            PyData::Vector(v) => v.clone().into_py(py),
            PyData::DataVector(v) => v.clone().into_py(py),
            PyData::Map(v) => v.clone().into_py(py),
        }
    }
}

/// Convert from `DataValue` to `PyData`
impl From<&DataValue> for PyData {
    fn from(value: &DataValue) -> Self {
        match value {
            DataValue::Real(v) => PyData::Real(*v),
            DataValue::Integer(v) => PyData::Integer(*v),
            DataValue::USize(v) => PyData::USize(*v),
            DataValue::Vector(v) => PyData::Vector(v.clone()),
            DataValue::DataVector(v) => {
                let data = v
                    .iter()
                    .map(|e| {
                        let a: PyData = e.into();
                        a
                    })
                    .collect::<Vec<PyData>>();
                PyData::DataVector(data)
            }
            DataValue::Map(v) => {
                let mut map: HashMap<String, PyData> = HashMap::new();
                for (name, e) in v {
                    let a: PyData = e.into();
                    map.insert(name.clone(), a);
                }
                PyData::Map(map)
            }
        }
    }
}

#[pyclass(name = "Individual")]
pub struct PyIndividual {
    #[pyo3(get)]
    constraints: HashMap<String, f64>,
    #[pyo3(get)]
    objectives: HashMap<String, f64>,
    #[pyo3(get)]
    constraint_violation: f64,
    #[pyo3(get)]
    is_feasible: bool,
    // private fields
    individual: Individual,
    variables: HashMap<String, VariableValue>,
    data: HashMap<String, DataValue>,
}

/// Convert `Individual` to `PyIndividual`
impl From<&Individual> for PyIndividual {
    fn from(value: &Individual) -> Self {
        PyIndividual {
            individual: value.clone(),
            variables: value.variables(),
            constraints: value.constraints(),
            objectives: value.objectives(),
            data: value.data().into(),
            constraint_violation: value.constraint_violation(),
            is_feasible: value.is_feasible(),
        }
    }
}

#[pymethods]
impl PyIndividual {
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Individual(variables={:?}, objectives={:?}, constraints={:?}, is_feasible={})",
            self.variables, self.objectives, self.constraints, self.is_feasible
        ))
    }

    pub fn __str__(&self) -> String {
        self.__repr__().unwrap()
    }

    pub fn get_objective_value(&self, name: String) -> PyResult<f64> {
        self.individual
            .get_objective_value(&name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn get_constraint_value(&self, name: String) -> PyResult<f64> {
        self.individual
            .get_constraint_value(&name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn get_variable_value(&self, name: String) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            self.individual
                .get_variable_value(&name)
                .map(|v| variable_value_to_py(v, py))
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[getter]
    pub fn variables(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut dict = HashMap::new();
            for (var_name, var) in &self.variables {
                dict.insert(var_name, variable_value_to_py(var, py));
            }
            Ok(dict.into_py(py))
        })
    }

    #[getter]
    pub fn data(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut dict: HashMap<&String, PyObject> = HashMap::new();
            for (name, value) in &self.data {
                let data: PyData = value.into();
                let py_data: PyObject = data.into_py(py);
                dict.insert(name, py_data);
            }
            Ok(dict.into_py(py))
        })
    }
}
