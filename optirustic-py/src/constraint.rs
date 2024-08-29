use std::fmt::{Display, Formatter};

use pyo3::prelude::*;

use optirustic::core::{Constraint, RelationalOperator};

/// Constraint
#[pyclass(name = "RelationalOperator", eq, eq_int)]
#[derive(PartialEq)]
pub enum PyRelationalOperator {
    EqualTo,
    NotEqualTo,
    LessOrEqualTo,
    LessThan,
    GreaterOrEqualTo,
    GreaterThan,
}

impl Display for PyRelationalOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            // PyObjectiveDirection::MAXIMISE => write!(f, "maximise"),
            PyRelationalOperator::EqualTo => write!(f, "=="),
            PyRelationalOperator::NotEqualTo => write!(f, "!="),
            PyRelationalOperator::LessOrEqualTo => write!(f, "<="),
            PyRelationalOperator::LessThan => write!(f, "<"),
            PyRelationalOperator::GreaterOrEqualTo => write!(f, ">="),
            PyRelationalOperator::GreaterThan => write!(f, ">"),
        }
    }
}

#[pyclass(name = "Constraint")]
pub struct PyConstraint {
    name: String,
    operator: PyRelationalOperator,
    target: f64,
}

#[pymethods]
impl PyConstraint {
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Constraint(name='{}', operator='{}', target={})",
            self.name, self.operator, self.target
        ))
    }

    pub fn __str__(&self) -> String {
        self.__repr__().unwrap()
    }
}

impl From<&Constraint> for PyConstraint {
    fn from(value: &Constraint) -> Self {
        let operator = match &value.operator() {
            RelationalOperator::EqualTo => PyRelationalOperator::EqualTo,
            RelationalOperator::NotEqualTo => PyRelationalOperator::NotEqualTo,
            RelationalOperator::LessOrEqualTo => PyRelationalOperator::LessOrEqualTo,
            RelationalOperator::LessThan => PyRelationalOperator::LessThan,
            RelationalOperator::GreaterOrEqualTo => PyRelationalOperator::GreaterOrEqualTo,
            RelationalOperator::GreaterThan => PyRelationalOperator::GreaterThan,
        };
        PyConstraint {
            name: value.name(),
            operator,
            target: value.target(),
        }
    }
}
