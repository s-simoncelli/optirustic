use std::collections::HashMap;

use serde::{Deserialize, Serialize, Serializer};

use crate::core::OError;

/// The data type and value that can be stored in an individual or algorithm..
#[derive(Clone, Deserialize, Debug)]
#[serde(untagged)]
pub enum DataValue {
    /// The value for a floating-point number. This is a f64.
    Real(f64),
    /// The value for an integer number. This is an i64.
    Integer(i64),
    /// The value for an usize.
    USize(usize),
    /// The value for a vector of floating-point numbers.
    Vector(Vec<f64>),
    /// The value for a vector of nested data.
    DataVector(Vec<DataValue>),
    /// The value for a Hashmap
    Map(HashMap<String, DataValue>),
}

impl Serialize for DataValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            DataValue::Real(v) => serializer.serialize_f64(*v),
            DataValue::Integer(v) => serializer.serialize_i64(*v),
            DataValue::USize(v) => serializer.serialize_u64(*v as u64),
            DataValue::Vector(v) => serializer.collect_seq(v),
            DataValue::DataVector(v) => serializer.collect_seq(v),
            DataValue::Map(v) => serializer.collect_map(v),
        }
    }
}

impl PartialEq for DataValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DataValue::Real(s), DataValue::Real(o)) => (s.is_nan() && o.is_nan()) || (*s == *o),
            (DataValue::Integer(s), DataValue::Integer(o)) => *s == *o,
            (DataValue::USize(s), DataValue::USize(o)) => s == o,
            (DataValue::Vector(s), DataValue::Vector(o)) => s == o,
            _ => false,
        }
    }
}

impl DataValue {
    /// Get the value if the data is of real type. This returns an error if the data is not real.
    ///
    /// returns: `Result<f64, OError>`
    pub fn as_real(&self) -> Result<f64, OError> {
        if let DataValue::Real(v) = self {
            Ok(*v)
        } else {
            Err(OError::WrongDataType("real".to_string()))
        }
    }

    /// Get the value if the data is of integer type. This returns an error if the data is not an
    /// integer.
    ///
    /// returns: `Result<f64, OError>`
    pub fn as_integer(&self) -> Result<i64, OError> {
        if let DataValue::Integer(v) = self {
            Ok(*v)
        } else {
            Err(OError::WrongDataType("integer".to_string()))
        }
    }

    /// Get the value if the data is of vector of f64. This returns an error if the data is not a
    /// vector.
    ///
    /// returns: `Result<&Vec<f64, OError>`
    pub fn as_f64_vec(&self) -> Result<&Vec<f64>, OError> {
        if let DataValue::Vector(v) = self {
            Ok(v)
        } else {
            Err(OError::WrongDataType("vector of f64".to_string()))
        }
    }
    /// Get the mutable value if the data is of vector of f64. This returns an error if the data
    /// is not a vector.
    ///
    /// returns: `Result<&mut Vec<f64, OError>`
    pub fn as_mut_f64_vec(&mut self) -> Result<&mut Vec<f64>, OError> {
        if let DataValue::Vector(v) = self {
            Ok(v)
        } else {
            Err(OError::WrongDataType("vector of f64".to_string()))
        }
    }

    /// Get the value if the data is of vector of data. This returns an error if the data is not a
    /// data vector.
    ///
    /// returns: `Result<&Vec<DataValue>, OError>`
    pub fn as_data_vec(&self) -> Result<&Vec<DataValue>, OError> {
        if let DataValue::DataVector(v) = self {
            Ok(v)
        } else {
            Err(OError::WrongDataType("vector of data".to_string()))
        }
    }

    /// Get the mutable value if the data is of vector of data. This returns an error if the data
    /// is not a data vector.
    ///
    /// returns: `Result<&mut Vec<DataValue>, OError>`
    pub fn as_mut_data_vec(&mut self) -> Result<&mut Vec<DataValue>, OError> {
        if let DataValue::DataVector(v) = self {
            Ok(v)
        } else {
            Err(OError::WrongDataType("vector of data".to_string()))
        }
    }

    /// Get the value if the data is a mao. This returns an error if the data is not a map.
    ///
    /// returns: `Result<HashMap<String, DataValue>, OError>`
    pub fn as_map(&self) -> Result<&HashMap<String, DataValue>, OError> {
        if let DataValue::Map(v) = self {
            Ok(v)
        } else {
            Err(OError::WrongDataType("map".to_string()))
        }
    }

    /// Get the value if the data is of usize type. This returns an error if the data is not an
    /// usize.
    ///
    /// returns: `Result<f64, OError>`
    pub fn as_usize(&self) -> Result<usize, OError> {
        if let DataValue::USize(v) = self {
            Ok(*v)
        } else {
            Err(OError::WrongDataType("usize".to_string()))
        }
    }
}
