use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use rand::distributions::uniform::SampleUniform;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::core::{Individual, OError, Problem};

/// A trait to define a decision variable.
pub trait Variable<T>: Display {
    /// Generate a new random value for the variable.
    fn generate(&self) -> T;
    /// Get the variable name
    fn name(&self) -> String;
}

pub trait BoundedNumberTrait: SampleUniform + PartialOrd + Display + Clone {}
impl<T: SampleUniform + PartialOrd + Display + Clone> BoundedNumberTrait for T {}

/// A number between a lower and upper bound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundedNumber<N: BoundedNumberTrait> {
    /// The variable name
    name: String,
    /// The minimum value bound.
    min_value: N,
    /// The maximum value bound.
    max_value: N,
}

impl<N: BoundedNumberTrait> BoundedNumber<N> {
    /// Create a new decision variable using a number bounded between a lower and upper bound.
    /// When a new value `N` for this variable is generated, the new value will be picked such that
    /// `min_value` <= `N` <= `max_value`.
    ///
    /// # Arguments
    ///
    /// * `name`: The variable name.
    /// * `min_value`: The lower bound.
    /// * `max_value`: The upper bound.
    ///
    /// returns: `Result<BoundedNumber, OError>`
    pub fn new(name: &str, min_value: N, max_value: N) -> Result<Self, OError> {
        if min_value >= max_value {
            return Err(OError::TooLargeLowerBound(
                min_value.to_string(),
                max_value.to_string(),
            ));
        }
        Ok(Self {
            name: name.to_string(),
            min_value,
            max_value,
        })
    }

    /// The variable lower bound.
    ///
    /// return: `N`
    pub fn min_value(&self) -> N {
        self.min_value.clone()
    }

    /// The variable upper bound.
    ///
    /// return: `N`
    pub fn max_value(&self) -> N {
        self.max_value.clone()
    }

    /// The variable upper and lower bound.
    ///
    /// return: `N`
    pub fn bounds(&self) -> (N, N) {
        (self.min_value.clone(), self.max_value.clone())
    }
}

impl<N: BoundedNumberTrait + Copy> Variable<N> for BoundedNumber<N> {
    /// Randomly generate a new bounded number.
    fn generate(&self) -> N {
        let mut rng = rand::thread_rng();
        rng.gen_range(self.min_value..=self.max_value)
    }
    fn name(&self) -> String {
        self.name.clone()
    }
}

impl<N: BoundedNumberTrait> Display for BoundedNumber<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BoundedNumber '{}' to [{}; {}]",
            self.name, self.min_value, self.max_value
        )
    }
}

/// A boolean variable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Boolean {
    /// The variable name.
    name: String,
}

impl Boolean {
    /// Create a new boolean variable.
    ///
    /// # Arguments
    ///
    /// * `name`: The variable name.
    ///
    /// returns: `Boolean`
    pub fn new(name: &str) -> Self {
        Boolean {
            name: name.to_string(),
        }
    }
}
impl Display for Boolean {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Boolean '{}'", self.name)
    }
}

impl Variable<bool> for Boolean {
    /// Randomly generate a boolean value.
    ///
    /// return: `bool`
    fn generate(&self) -> bool {
        let mut rng = rand::thread_rng();
        !matches!([0, 1].choose(&mut rng).unwrap(), 0)
    }
    fn name(&self) -> String {
        self.name.clone()
    }
}

/// A variable choice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    /// The variable name.
    name: String,
    /// The list of choices.
    choices: Vec<String>,
}

impl Choice {
    /// Create a new list of choices.
    ///
    /// # Arguments
    ///
    /// * `name`: The variable name.
    /// * `choices`: The list of choices.
    ///
    /// returns: `Choice`
    pub fn new(name: &str, choices: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            choices,
        }
    }
}

impl Display for Choice {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Choice '{}': {}", self.name, self.choices.join(", "))
    }
}

impl Variable<String> for Choice {
    /// Randomly pick a choice.
    fn generate(&self) -> String {
        let mut rng = rand::thread_rng();
        let choice_index = (0..self.choices.len()).choose(&mut rng).unwrap();
        self.choices[choice_index].clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

/// The types of variables to set on a problem.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VariableType {
    /// A continuous bounded variable (f64)
    Real(BoundedNumber<f64>),
    /// A discrete bounded variable (i64)
    Integer(BoundedNumber<i64>),
    /// A boolean variable
    Boolean(Boolean),
    /// A variable representing a choice (as string)
    Choice(Choice),
}

impl VariableType {
    /// Generate a new random variable value based on its type.
    ///
    /// returns: `VariableValue`
    pub fn generate_random_value(&self) -> VariableValue {
        match &self {
            VariableType::Real(v) => VariableValue::Real(v.generate()),
            VariableType::Integer(v) => VariableValue::Integer(v.generate()),
            VariableType::Boolean(v) => VariableValue::Boolean(v.generate()),
            VariableType::Choice(v) => VariableValue::Choice(v.generate()),
        }
    }

    /// Get the variable name.
    ///
    /// return: `String`
    pub fn name(&self) -> String {
        match self {
            VariableType::Real(t) => t.name.clone(),
            VariableType::Integer(t) => t.name.clone(),
            VariableType::Boolean(t) => t.name.clone(),
            VariableType::Choice(t) => t.name.clone(),
        }
    }

    /// Check if the variable is a real number.
    ///
    /// return: `bool`
    pub(crate) fn is_real(&self) -> bool {
        matches!(self, VariableType::Real(_))
    }

    /// Check if the variable is an integer number.
    ///
    /// return: `bool`
    pub(crate) fn is_integer(&self) -> bool {
        matches!(self, VariableType::Integer(_))
    }
}

impl Display for VariableType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VariableType::Real(v) => write!(f, "{v}").unwrap(),
            VariableType::Integer(v) => write!(f, "{v}").unwrap(),
            VariableType::Boolean(v) => write!(f, "{v}").unwrap(),
            VariableType::Choice(v) => write!(f, "{v}").unwrap(),
        };
        Ok(())
    }
}

/// A trait to allow generating a value for an individual.
pub trait VariableValueGenerator {
    fn generate(individual: &Individual);
}

/// The value of a variable to set on an individual.
#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VariableValue {
    /// The value for a floating-point number. This is a f64.
    Real(f64),
    /// The value for an integer number. This is an i64.
    Integer(i64),
    /// The value for a boolean variable.
    Boolean(bool),
    /// The value for a choice variable.
    Choice(String),
}

impl PartialEq for VariableValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (VariableValue::Real(s), VariableValue::Real(o)) => {
                (s.is_nan() && o.is_nan()) || (*s == *o)
            }
            (VariableValue::Integer(s), VariableValue::Integer(o)) => *s == *o,
            (VariableValue::Boolean(s), VariableValue::Boolean(o)) => s == o,
            (VariableValue::Choice(s), VariableValue::Choice(o)) => s == o,
            _ => false,
        }
    }
}

impl VariableValue {
    /// Check if the variable value matches the variable type set on the problem. This return an
    /// error if the variable name does not exist in the problem.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the variable in the problem.
    /// * `problem`: The problem being solved.
    ///
    /// returns: `Result<bool, OError>`
    pub fn match_type(&self, name: &str, problem: Arc<Problem>) -> Result<bool, OError> {
        let value = match problem.get_variable(name)? {
            VariableType::Real(_) => matches!(self, VariableValue::Real(_)),
            VariableType::Integer(_) => matches!(self, VariableValue::Integer(_)),
            VariableType::Boolean(_) => matches!(self, VariableValue::Boolean(_)),
            VariableType::Choice(_) => matches!(self, VariableValue::Choice(_)),
        };
        Ok(value)
    }

    /// Get the value if the variable is of real type. This returns an error if the variable is not
    /// real.
    ///
    /// returns: `Result<f64, OError>`
    pub fn as_real(&self) -> Result<f64, OError> {
        if let VariableValue::Real(v) = self {
            Ok(*v)
        } else {
            Err(OError::WrongVariableType("real".to_string()))
        }
    }

    /// Get the value if the variable is of discrete type. This returns an error if the variable is
    /// not an integer.
    ///
    /// returns: `Result<f64, OError>`
    pub fn as_integer(&self) -> Result<i64, OError> {
        if let VariableValue::Integer(v) = self {
            Ok(*v)
        } else {
            Err(OError::WrongVariableType("integer".to_string()))
        }
    }
}

impl VariableValueGenerator for VariableValue {
    fn generate(_individual: &Individual) {
        // TODO loop variables
        todo!()
    }
}

impl Debug for VariableValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VariableValue::Real(v) => write!(f, "{v}").unwrap(),
            VariableValue::Integer(v) => write!(f, "{v}").unwrap(),
            VariableValue::Boolean(v) => write!(f, "{v}").unwrap(),
            VariableValue::Choice(v) => write!(f, "{v}").unwrap(),
        };
        Ok(())
    }
}
