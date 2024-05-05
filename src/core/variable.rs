use std::fmt;
use std::fmt::{Debug, Display, Formatter};

use rand::distributions::uniform::SampleUniform;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;

use crate::core::Problem;

/// The trait for a decision variable.
pub trait Variable<T>: Display {
    /// Generate a new random value for the variable.
    fn generate(&self) -> T;
    /// Get the variable name
    fn name(&self) -> String;
}

pub trait BoundedNumberTrait: SampleUniform + PartialOrd + Display {}
impl<T: SampleUniform + PartialOrd + Display> BoundedNumberTrait for T {}

/// A number between a lower and upper bound.
#[derive(Clone, Debug)]
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
    /// returns: `Result<BoundedNumber, String>`
    pub fn new(name: &str, min_value: N, max_value: N) -> Result<Self, String> {
        if min_value >= max_value {
            return Err(format!(
                "The min value ({}) must be strictly smaller than the max value ({}).",
                min_value, max_value
            ));
        }
        Ok(Self {
            name: name.to_string(),
            min_value,
            max_value,
        })
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
    /// returns: `Result<Choice, String>`
    pub fn new(name: &str, choices: Vec<String>) -> Result<Self, String> {
        Ok(Self {
            name: name.to_string(),
            choices,
        })
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
#[derive(Clone, Debug)]
pub enum VariableType {
    F64(BoundedNumber<f64>),
    F32(BoundedNumber<f32>),
    I64(BoundedNumber<i64>),
    I32(BoundedNumber<i32>),
    U64(BoundedNumber<u64>),
    U32(BoundedNumber<u32>),
    Boolean(Boolean),
    Choice(Choice),
}

impl VariableType {
    /// Get the initial value to set for a variable type.
    ///
    /// return: `VariableValue`.
    pub(crate) fn initial_value(&self) -> VariableValue {
        match self {
            VariableType::F64(t) => VariableValue::F64(t.min_value),
            VariableType::F32(t) => VariableValue::F32(t.min_value),
            VariableType::I64(t) => VariableValue::I64(t.min_value),
            VariableType::I32(t) => VariableValue::I32(t.min_value),
            VariableType::U64(t) => VariableValue::U64(t.min_value),
            VariableType::U32(t) => VariableValue::U32(t.min_value),
            VariableType::Boolean(_) => VariableValue::Boolean(false),
            VariableType::Choice(_) => VariableValue::Choice("NA".to_string()),
        }
    }
}

impl Display for VariableType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VariableType::F64(v) => write!(f, "{v}").unwrap(),
            VariableType::F32(v) => write!(f, "{v}").unwrap(),
            VariableType::I64(v) => write!(f, "{v}").unwrap(),
            VariableType::I32(v) => write!(f, "{v}").unwrap(),
            VariableType::U64(v) => write!(f, "{v}").unwrap(),
            VariableType::U32(v) => write!(f, "{v}").unwrap(),
            VariableType::Boolean(v) => write!(f, "{v}").unwrap(),
            VariableType::Choice(v) => write!(f, "{v}").unwrap(),
        };
        Ok(())
    }
}

/// The value of a variable to set on an individual.
pub enum VariableValue {
    /// The value for a f64 variable.
    F64(f64),
    /// The value for a f32 variable.
    F32(f32),
    /// The value for an i64 variable.
    I64(i64),
    /// The value for an i32 variable.
    I32(i32),
    /// The value for an u64 variable.
    U64(u64),
    /// The value for an u32 variable.
    U32(u32),
    /// The value for a boolean variable.
    Boolean(bool),
    /// The value for a choice variable.
    Choice(String),
}

impl VariableValue {
    /// Generate a new random variable value.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the variable in the problem.
    /// * `problem`: The problem being solved.
    ///
    /// returns: `VariableValue`
    pub(crate) fn generate_random(
        &self,
        name: &str,
        problem: &Problem,
    ) -> Result<VariableValue, String> {
        let value = match problem.get_variable(name)? {
            VariableType::F64(v) => VariableValue::F64(v.generate()),
            VariableType::F32(v) => VariableValue::F32(v.generate()),
            VariableType::I64(v) => VariableValue::I64(v.generate()),
            VariableType::I32(v) => VariableValue::I32(v.generate()),
            VariableType::U64(v) => VariableValue::U64(v.generate()),
            VariableType::U32(v) => VariableValue::U32(v.generate()),
            VariableType::Boolean(v) => VariableValue::Boolean(v.generate()),
            VariableType::Choice(v) => VariableValue::Choice(v.generate()),
        };
        Ok(value)
    }
}
impl Debug for VariableValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VariableValue::F64(v) => write!(f, "{v}").unwrap(),
            VariableValue::F32(v) => write!(f, "{v}").unwrap(),
            VariableValue::I64(v) => write!(f, "{v}").unwrap(),
            VariableValue::I32(v) => write!(f, "{v}").unwrap(),
            VariableValue::U64(v) => write!(f, "{v}").unwrap(),
            VariableValue::U32(v) => write!(f, "{v}").unwrap(),
            VariableValue::Boolean(v) => write!(f, "{v}").unwrap(),
            VariableValue::Choice(v) => write!(f, "{v}").unwrap(),
        };
        Ok(())
    }
}
