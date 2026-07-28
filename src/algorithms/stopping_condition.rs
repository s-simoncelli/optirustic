use serde::{Deserialize, Serialize};
use std::{fmt::Display, sync::Arc};

/// A trait to use to define a custom stopping condition function.
pub trait CustomStoppingCondition: Sync + Send {
    /// Define a unique name to the condition.
    fn name(&self) -> String;

    /// Return `true` to stop the algorithm, `false` otherwise`
    fn is_met(&self) -> bool;
}

/// The type of stopping condition. Pick one type to inform the algorithm how/when it should
/// terminate the population evolution.
#[derive(Serialize, Deserialize, Clone)]
pub enum StoppingCondition {
    /// Set a maximum duration (as number of minutes).
    MaxDurationAsMinutes(u32),
    /// Set a maximum duration (as number of hours).
    MaxDurationAsHours(u32),
    /// Set a maximum number of generations.
    MaxGeneration(u32),
    /// Set a maximum number of function evaluations.
    MaxFunctionEvaluations(u32),
    /// Stop when at least on condition is met (this acts as an OR operator).
    Any(Vec<StoppingCondition>),
    /// Stop when all conditions are met (this acts as an AND operator).
    All(Vec<StoppingCondition>),
    #[serde(skip_serializing, skip_deserializing)]
    Function(Arc<dyn CustomStoppingCondition>),
}

impl StoppingCondition {
    /// A name describing the stopping condition.
    ///
    /// returns: `String`
    pub fn name(&self) -> String {
        match self {
            StoppingCondition::MaxDurationAsMinutes(v) => format!("maximum duration={v} minutes"),
            StoppingCondition::MaxDurationAsHours(v) => format!("maximum duration={v} hours"),
            StoppingCondition::MaxGeneration(v) => format!("maximum number of generations={v}"),
            StoppingCondition::MaxFunctionEvaluations(v) => {
                format!("maximum number of function evaluations={v}")
            }
            StoppingCondition::Any(s) => s
                .iter()
                .map(|cond| cond.name())
                .collect::<Vec<String>>()
                .join(" OR "),
            StoppingCondition::All(s) => s
                .iter()
                .map(|cond| cond.name())
                .collect::<Vec<String>>()
                .join(" AND "),
            StoppingCondition::Function(custom_stopping_condition) => {
                format!("Custom condition {}", custom_stopping_condition.name())
            }
        }
    }

    /// Check whether the stopping condition is a vector and has nested vector in it.
    ///
    /// # Arguments
    ///
    /// * `conditions`: A vector of stopping conditions.
    ///
    /// returns: `bool`
    pub fn has_nested_vector(conditions: &[StoppingCondition]) -> bool {
        conditions.iter().any(|c| match c {
            StoppingCondition::Any(_) | StoppingCondition::All(_) => true,
            _ => false,
        })
    }
}

impl Display for StoppingCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            StoppingCondition::MaxDurationAsMinutes(duration) => write!(f, "{duration} minutes"),
            StoppingCondition::MaxDurationAsHours(duration) => write!(f, "{duration} hours"),
            StoppingCondition::MaxGeneration(generation) => write!(f, "{generation} generations"),
            StoppingCondition::MaxFunctionEvaluations(nfe) => write!(f, "{nfe} evaluations"),
            StoppingCondition::Any(values) => {
                let values: Vec<String> = values.iter().map(|c| format!("{c}")).collect();
                write!(f, "{}", values.join(" OR "))
            }
            StoppingCondition::All(values) => {
                let values: Vec<String> = values.iter().map(|c| format!("{c}")).collect();
                write!(f, "{}", values.join(" AND "))
            }
            StoppingCondition::Function(custom_stopping_condition) => {
                write!(f, "c{}", custom_stopping_condition.name())
            }
        }
    }
}

#[cfg(feature = "python")]
pub mod py {
    use crate::algorithms::StoppingCondition;
    use pyo3::{prelude::*, types::PyList, IntoPyObjectExt};

    /// Handle conversion of `StoppingCondition` into Python object for `NSGA*Args` structs.
    impl<'py> IntoPyObject<'py> for StoppingCondition {
        type Target = PyAny;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                StoppingCondition::MaxDurationAsMinutes(d) => {
                    PyStoppingConditionValue::max_duration_as_minutes(d).into_bound_py_any(py)
                }
                StoppingCondition::MaxDurationAsHours(d) => {
                    PyStoppingConditionValue::max_duration_as_hours(d).into_bound_py_any(py)
                }
                StoppingCondition::MaxGeneration(g) => {
                    PyStoppingConditionValue::max_generation(g).into_bound_py_any(py)
                }
                StoppingCondition::MaxFunctionEvaluations(nfe) => {
                    PyStoppingConditionValue::max_function_evaluations(nfe).into_bound_py_any(py)
                }
                StoppingCondition::Any(stopping_conditions)
                | StoppingCondition::All(stopping_conditions) => {
                    let items = stopping_conditions
                        .into_iter()
                        .map(|c| c.into_pyobject(py))
                        .collect::<Result<Vec<_>, _>>()?;
                    PyList::new(py, items)?.into_bound_py_any(py)
                }
                // StoppingCondition::All(stopping_conditions) => stopping_conditions.iter().map(|c|c.into_pyobject(py)).collect()?,
                StoppingCondition::Function(_) => {
                    panic!("Function stopping condition not supported")
                } // _ => panic!("Function stopping condition not supported"),
            }
            // out.into_bound_py_any(py)
        }
    }

    /// The stopping condition class in Python. Each enum item is a Python function of the
    /// [`StoppingConditionValue`] class. Items are lower-case to be PEP compliant.
    #[pyclass(name = "StoppingCondition", from_py_object)]
    #[derive(Clone)]
    #[allow(non_camel_case_types)]
    pub enum PyStoppingConditionValue {
        max_duration_as_minutes(u32),
        max_duration_as_hours(u32),
        max_generation(u32),
        max_function_evaluations(u32),
    }

    #[pymethods]
    impl PyStoppingConditionValue {
        fn value(&self) -> u32 {
            match self {
                PyStoppingConditionValue::max_duration_as_minutes(v) => *v,
                PyStoppingConditionValue::max_duration_as_hours(v) => *v,
                PyStoppingConditionValue::max_generation(v) => *v,
                PyStoppingConditionValue::max_function_evaluations(v) => *v,
            }
        }

        fn __repr__(&self) -> PyResult<String> {
            let attr = match &self {
                PyStoppingConditionValue::max_duration_as_minutes(duration) => {
                    format!("duration={duration} minutes")
                }
                PyStoppingConditionValue::max_duration_as_hours(duration) => {
                    format!("duration={duration} hours")
                }
                PyStoppingConditionValue::max_generation(generation) => {
                    format!("generation={generation} generations")
                }
                PyStoppingConditionValue::max_function_evaluations(nfe) => {
                    format!("NFE={nfe} evaluations")
                }
            };
            Ok(format!("StoppingConditionValue({attr})"))
        }

        fn __str__(&self) -> String {
            self.__repr__().unwrap()
        }
    }

    /// Allow conversion to [`StoppingCondition`] from Python when an algorithm is initialised
    impl From<PyStoppingConditionValue> for StoppingCondition {
        fn from(cond: PyStoppingConditionValue) -> Self {
            match cond {
                PyStoppingConditionValue::max_duration_as_minutes(duration) => {
                    StoppingCondition::MaxDurationAsMinutes(duration)
                }
                PyStoppingConditionValue::max_duration_as_hours(duration) => {
                    StoppingCondition::MaxDurationAsHours(duration)
                }
                PyStoppingConditionValue::max_generation(generation) => {
                    StoppingCondition::MaxGeneration(generation)
                }
                PyStoppingConditionValue::max_function_evaluations(nfe) => {
                    StoppingCondition::MaxFunctionEvaluations(nfe)
                }
            }
        }
    }

    /// Python conversion of stopping condition or list of.
    #[derive(FromPyObject)]
    pub enum PyStoppingConditionMap {
        #[pyo3(transparent, annotation = "condition")]
        Condition(PyStoppingConditionValue),
        #[pyo3(transparent, annotation = "list of conditions")]
        Vector(Vec<PyStoppingConditionValue>),
    }

    /// Handle initialisation of condition(s) from Python to Rust/
    impl From<PyStoppingConditionMap> for StoppingCondition {
        fn from(value: PyStoppingConditionMap) -> Self {
            match value {
                PyStoppingConditionMap::Condition(condition) => condition.into(),
                PyStoppingConditionMap::Vector(conditions) => {
                    StoppingCondition::Any(conditions.into_iter().map(|c| c.into()).collect())
                }
            }
        }
    }
}
