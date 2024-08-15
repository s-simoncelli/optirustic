use pyo3::prelude::*;

use optirustic::core::VariableType;

#[pyclass(name = "VariableType", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum PyVariableType {
    Real,
    Integer,
    Boolean,
    Choice,
}

#[pymethods]
impl PyVariableType {
    pub fn __repr__(&self) -> String {
        let label = match &self {
            PyVariableType::Real => "real",
            PyVariableType::Integer => "integer",
            PyVariableType::Boolean => "boolean",
            PyVariableType::Choice => "choice",
        };
        label.to_string()
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Variable
#[pyclass(get_all, name = "Variable")]
pub struct PyVariable {
    name: String,
    var_type: PyVariableType,
    min_value: Option<f64>,
    max_value: Option<f64>,
}

/// Convert `VariableType` to `PyVariable`
impl From<&VariableType> for PyVariable {
    fn from(value: &VariableType) -> Self {
        let var_type = match value {
            VariableType::Real(_) => PyVariableType::Real,
            VariableType::Integer(_) => PyVariableType::Integer,
            VariableType::Boolean(_) => PyVariableType::Boolean,
            VariableType::Choice(_) => PyVariableType::Choice,
        };
        let (min_value, max_value) = match value {
            VariableType::Real(n) => (Some(n.min_value()), Some(n.max_value())),
            VariableType::Integer(n) => (Some(n.min_value() as f64), Some(n.max_value() as f64)),
            VariableType::Boolean(_) => (None, None),
            VariableType::Choice(_) => (None, None),
        };

        PyVariable {
            name: value.name(),
            var_type,
            min_value,
            max_value,
        }
    }
}

#[pymethods]
impl PyVariable {
    pub fn __repr__(&self) -> PyResult<String> {
        let args = if let (Some(min_value), Some(max_value)) = (self.min_value, self.max_value) {
            format!(", min-value={min_value}, max_value={max_value}")
        } else {
            String::from("")
        };
        Ok(format!(
            "Variable(name='{}', type='{}'{args})",
            self.name,
            self.var_type.__repr__()
        ))
    }

    pub fn __str__(&self) -> String {
        self.__repr__().unwrap()
    }
}
