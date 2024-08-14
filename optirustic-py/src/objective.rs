use std::fmt::{Display, Formatter};

use pyo3::prelude::*;

use optirustic::core::{Objective, ObjectiveDirection};

#[pyclass(name = "ObjectiveDirection", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum PyObjectiveDirection {
    Minimise,
    Maximise,
}

impl Display for PyObjectiveDirection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PyObjectiveDirection::Minimise => write!(f, "minimise"),
            PyObjectiveDirection::Maximise => write!(f, "maximise"),
        }
    }
}

#[pyclass(get_all, name = "Objective")]
pub struct PyObjective {
    name: String,
    direction: PyObjectiveDirection,
}

#[pymethods]
impl PyObjective {
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Objective(name='{}', direction='{}')",
            self.name, self.direction
        ))
    }

    pub fn __str__(&self) -> String {
        self.__repr__().unwrap()
    }
}

impl From<&Objective> for PyObjective {
    fn from(value: &Objective) -> Self {
        let direction = match value.direction() {
            ObjectiveDirection::Minimise => PyObjectiveDirection::Minimise,
            ObjectiveDirection::Maximise => PyObjectiveDirection::Maximise,
        };
        PyObjective {
            name: value.name(),
            direction,
        }
    }
}
