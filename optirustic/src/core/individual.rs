use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::ops::RangeBounds;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::core::{OError, Problem, VariableValue};
use crate::utils::hasmap_eq_with_nans;

/// The data type and value that can be stored in an individual's data.
#[derive(Clone, Debug)]
pub enum DataValue {
    /// The value for a floating-point number. This is a f64.
    Real(f64),
    /// The value for an integer number. This is an i64.
    Integer(i64),
    /// The value for an usize.
    USize(usize),
    /// The value for a vector of floating-point numbers.
    Vector(Vec<f64>),
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

    /// Get the value if the data is of vector type. This returns an error if the data is not a
    /// vector.
    ///
    /// returns: `Result<f64, OError>`
    pub fn as_vec(&self) -> Result<&Vec<f64>, OError> {
        if let DataValue::Vector(v) = self {
            Ok(v)
        } else {
            Err(OError::WrongDataType("vector".to_string()))
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

/// An individual in the population containing the problem solution, and the objective and
/// constraint values.
///
/// # Example
/// ```
/// use std::error::Error;
/// use optirustic::core::{BoundedNumber, Constraint, Individual, Problem, Objective,
/// ObjectiveDirection, RelationalOperator, EvaluationResult, Evaluator, VariableType, VariableValue};
/// use std::sync::Arc;
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
///
///     let var_types = vec![VariableType::Real(BoundedNumber::new("var1", 0.0, 2.0)?)];
///     let constraints = vec![Constraint::new("C1", RelationalOperator::EqualTo, 5.0)];
///
///     // dummy evaluator function
///     #[derive(Debug)]
///     struct UserEvaluator;
///     impl Evaluator for UserEvaluator {
///         fn evaluate(&self, _: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
///             Ok(EvaluationResult {
///                 constraints: Default::default(),
///                 objectives: Default::default(),
///             })
///         }
///     }
///     // create a new one-variable problem
///     let problem = Arc::new(Problem::new(objectives, var_types, Some(constraints), Box::new(UserEvaluator))?);
///
///     // create an individual and set the calculated variable
///     let mut a = Individual::new(problem.clone());
///     a.update_variable("var1", VariableValue::Real(0.2))?;
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Individual {
    /// The problem being solved
    problem: Arc<Problem>,
    /// The value of the problem variables for the individual.
    variable_values: HashMap<String, VariableValue>,
    /// The value of the constraints.
    constraint_values: HashMap<String, f64>,
    /// The values of the objectives.
    objective_values: HashMap<String, f64>,
    /// Whether the individual has been evaluated and the problem constraint and objective values
    /// are available. When an individual is created with some variables after the population
    /// evolves, constraints and objectives need to be evaluated using a user-defined function.
    evaluated: bool,
    /// Additional numeric data to store for the individuals (such as crowding distance or rank)
    /// depending on the algorithm the individuals are derived from.
    data: HashMap<String, DataValue>,
}

impl PartialEq for Individual {
    /// Compare two individual's constraints, variables, objectives and stored data.
    ///
    /// # Arguments
    ///
    /// * `other`: The other individual to compare.
    ///
    /// returns: `bool`
    fn eq(&self, other: &Self) -> bool {
        self.variable_values == other.variable_values
            && hasmap_eq_with_nans(&self.constraint_values, &other.constraint_values)
            && hasmap_eq_with_nans(&self.objective_values, &other.objective_values)
            && self.data == other.data
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IndividualExport {
    /// The value of the constraints.
    constraint_values: HashMap<String, f64>,
    /// The values of the objectives.
    objective_values: HashMap<String, f64>,
    /// The overall amount of violation of the solution constraints.
    constraint_violation: f64,
    /// The value of the problem variables for the individual.
    variable_values: HashMap<String, VariableValue>,
    /// Whether the solution meets all the problem constraints.
    is_feasible: bool,
}

impl Display for Individual {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Individual(variables={:?}, objectives={:?},constraints={:?})",
            self.variable_values, self.objective_values, self.constraint_values,
        )
    }
}

impl Individual {
    /// Create a new individual. An individual contains the solution after an evolution.
    ///
    /// # Arguments
    ///
    /// * `problem`: The problem being solved.
    ///
    /// returns: `Individual`
    pub fn new(problem: Arc<Problem>) -> Self {
        let mut variable_values: HashMap<String, VariableValue> = HashMap::new();
        for (variable_name, var_type) in problem.variables() {
            variable_values.insert(variable_name, var_type.generate_random_value());
        }

        let mut objective_values: HashMap<String, f64> = HashMap::new();
        for objective_name in problem.objective_names() {
            objective_values.insert(objective_name, f64::NAN);
        }

        let mut constraint_values: HashMap<String, f64> = HashMap::new();
        for constraint_name in problem.constraint_names() {
            constraint_values.insert(constraint_name, f64::NAN);
        }

        Self {
            problem,
            variable_values,
            constraint_values,
            objective_values,
            evaluated: false,
            data: HashMap::new(),
        }
    }

    /// Get the problem being solved with the individual.
    ///
    /// return `Arc<Problem>`
    pub fn problem(&self) -> Arc<Problem> {
        self.problem.clone()
    }

    /// Clone an individual by preserving only its solutions.
    ///
    /// return: `Individual`
    pub(crate) fn clone_variables(&self) -> Self {
        let mut i = Self::new(self.problem.clone());
        for (var_name, var_value) in self.variable_values.iter() {
            i.update_variable(var_name, var_value.clone()).unwrap()
        }
        i
    }

    /// Update the variable for a solution. This returns an error if the variable name does not
    /// exist or the variable value does not match the variable type set on the problem (for
    /// example [`VariableValue::Integer`] is provided but the type is [`crate::core::VariableType::Real`]).
    ///
    /// # Arguments
    ///
    /// * `name`: The variable to update.
    /// * `value`: The value to set.
    ///
    /// returns: `Result<(), OError>`
    pub fn update_variable(&mut self, name: &str, value: VariableValue) -> Result<(), OError> {
        if !self.variable_values.contains_key(name) {
            return Err(OError::NonExistingName(
                "variable".to_string(),
                name.to_string(),
            ));
        }
        if !value.match_type(name, self.problem.clone())? {
            return Err(OError::NonMatchingVariableType(name.to_string()));
        }
        if let Some(x) = self.variable_values.get_mut(name) {
            *x = value;
        }
        Ok(())
    }

    /// Update the objective for a solution. The value is saved as negative if the objective being
    /// updated is being maximised. This returns an error if the name does not exist in the problem.
    ///
    /// # Arguments
    ///
    /// * `name`: The objective to update.
    /// * `value`: The value to set.
    ///
    /// returns: `Result<(), OError>`
    pub fn update_objective(&mut self, name: &str, value: f64) -> Result<(), OError> {
        if !self.objective_values.contains_key(name) {
            return Err(OError::NonExistingName(
                "objective".to_string(),
                name.to_string(),
            ));
        }
        if value.is_nan() {
            return Err(OError::NaN("objective".to_string(), name.to_string()));
        }

        // invert the sign for maximisation problems
        let sign = match self.problem.is_objective_minimised(name)? {
            true => 1.0,
            false => -1.0,
        };
        if let Some(x) = self.objective_values.get_mut(name) {
            *x = sign * value;
        }
        Ok(())
    }

    /// Update a constraint.
    ///
    /// # Arguments
    ///
    /// * `name`: The constraint to update.
    /// * `value`: The value to set.
    ///
    /// returns: `Result<(), OError>`
    pub(crate) fn update_constraint(&mut self, name: &str, value: f64) -> Result<(), OError> {
        if !self.constraint_values.contains_key(name) {
            return Err(OError::NonExistingName(
                "constrain".to_string(),
                name.to_string(),
            ));
        }
        if value.is_nan() {
            return Err(OError::NaN("constraint".to_string(), name.to_string()));
        }
        if let Some(x) = self.constraint_values.get_mut(name) {
            *x = value;
        }
        Ok(())
    }

    /// Calculate the overall amount of violation of the solution constraints. This is a measure
    /// about how close (or far) the individual meets the constraints. If the solution is feasible,
    /// then the violation is 0.0. Otherwise, a positive number is returned.
    ///
    /// return: `f64`
    pub fn constraint_violation(&self) -> f64 {
        return self
            .problem
            .constraints()
            .iter()
            .map(|(name, c)| c.constraint_violation(self.constraint_values[name]))
            .sum();
    }

    /// Return whether the solution meets all the problem constraints.
    ///
    /// return: `bool`
    pub fn is_feasible(&self) -> bool {
        for (name, constraint_value) in self.constraint_values.iter() {
            if !self.problem.constraint_names().contains(name) {
                continue;
            }
            if !self
                .problem
                .get_constraint(name)
                .unwrap()
                .is_met(*constraint_value)
            {
                return false;
            }
        }
        true
    }

    /// Ge all the variables.
    ///
    /// returns: `HashMap<String, VariableValue>`
    pub fn variables(&self) -> HashMap<String, VariableValue> {
        self.variable_values.clone()
    }

    /// Ge the variable value by name. This return an error if the variable name does not exist.
    ///
    /// # Arguments
    ///
    /// * `name`: The variable name.
    ///
    /// returns: `Result<&VariableValue, OError>`
    pub fn get_variable_value(&self, name: &str) -> Result<&VariableValue, OError> {
        if !self.variable_values.contains_key(name) {
            return Err(OError::NonExistingName(
                "variable".to_string(),
                name.to_string(),
            ));
        }

        Ok(&self.variable_values[name])
    }

    /// Ge the constraint value by name. This return an error if the constraint name does not exist.
    ///
    /// # Arguments
    ///
    /// * `name`: The constraint name.
    ///
    /// returns: `Result<f64, OError>`
    pub fn get_constraint_value(&self, name: &str) -> Result<f64, OError> {
        if !self.constraint_values.contains_key(name) {
            return Err(OError::NonExistingName(
                "constraint".to_string(),
                name.to_string(),
            ));
        }

        Ok(self.constraint_values[name])
    }

    /// Get the number stored in a real variable by name. This returns an error if the variable
    /// name does not exist or the variable is not of type real.
    ///
    /// # Arguments
    ///
    /// * `name`: The variable name.
    ///
    /// returns: `Result<f64, OError>`
    pub fn get_real_value(&self, name: &str) -> Result<f64, OError> {
        self.get_variable_value(name)?
            .as_real()
            .map_err(|_| OError::WrongVariableTypeWithName(name.to_string(), "real".to_string()))
    }

    /// Get the number stored in an integer variable by name. This returns an error if the variable
    /// name does not exist or the variable is not of type integer.
    ///
    /// # Arguments
    ///
    /// * `name`: The variable name.
    ///
    /// returns: `Result<i64, OError>`
    pub fn get_integer_value(&self, name: &str) -> Result<i64, OError> {
        self.get_variable_value(name)?
            .as_integer()
            .map_err(|_| OError::WrongVariableTypeWithName(name.to_string(), "integer".to_string()))
    }

    /// Ge the objective value by name. This returns an error if the objective does not exist.
    ///
    /// # Arguments
    ///
    /// * `name`: The objective name.
    ///
    /// returns: `Result<f64, OError>`
    pub fn get_objective_value(&self, name: &str) -> Result<f64, OError> {
        if !self.objective_values.contains_key(name) {
            return Err(OError::NonExistingName(
                "objective".to_string(),
                name.to_string(),
            ));
        }

        Ok(self.objective_values[name])
    }

    /// Ge the vector with the objective values for the individual. The size of the vector will
    /// equal the number of problem objectives.
    ///
    /// returns: `Result<Vec<f64>, OError>`
    pub fn get_objective_values(&self) -> Result<Vec<f64>, OError> {
        self.problem
            .objective_names()
            .iter()
            .map(|obj_name| self.get_objective_value(obj_name))
            .collect()
    }

    /// Ge the vector with the objective values for the individual and transform their value using
    /// a closure. The size of the vector will equal the number of problem objectives.
    ///
    /// # Arguments
    ///
    /// * `transform`: The function to apply to transform each objective value. This function
    /// receives the objective value and its name.
    ///
    /// returns: `Result<Vec<f64>, OError>`
    pub fn transform_objective_values<F: Fn(f64, String) -> Result<f64, OError>>(
        &self,
        transform: F,
    ) -> Result<Vec<f64>, OError> {
        self.problem
            .objective_names()
            .iter()
            .map(|obj_name| {
                let val = self.get_objective_value(obj_name)?;
                transform(val, obj_name.clone())
            })
            .collect()
    }

    /// Check if the individual was evaluated.
    ///
    /// return: `bool`
    pub fn is_evaluated(&self) -> bool {
        self.evaluated
    }

    /// Set the individual as evaluated. This means that its constraints and objectives have been
    /// calculated for its solution.
    pub fn set_evaluated(&mut self) {
        self.evaluated = true;
    }

    /// Store custom data on the individual.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the data.
    /// * `value`: The value.
    ///
    /// returns: `()`.
    pub fn set_data(&mut self, name: &str, value: DataValue) {
        self.data.insert(name.to_string(), value);
    }

    /// Get a copy of the custom data set on the individual. This returns an error if no custom
    /// data with the provided `name` is set on the individual.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the data.
    ///
    /// returns: `Result<DataValue, OError>`
    pub fn get_data(&self, name: &str) -> Result<DataValue, OError> {
        self.data
            .get(name)
            .cloned()
            .ok_or(OError::WrongDataName(name.to_string()))
    }

    /// Export all the solution data (constraint and objective values, constraint violation and
    /// feasibility).
    ///
    /// return: `IndividualExport`
    pub fn serialise(&self) -> IndividualExport {
        // invert maximised objective for user
        let mut objective_values = self.objective_values.clone();
        for name in self.problem.objective_names() {
            match self.problem.is_objective_minimised(&name) {
                Ok(is_minimised) => {
                    if !is_minimised {
                        *objective_values.get_mut(&name).unwrap() *= -1.0;
                    }
                }
                Err(_) => continue,
            }
        }

        IndividualExport {
            constraint_values: self.constraint_values.clone(),
            objective_values,
            constraint_violation: self.constraint_violation(),
            variable_values: self.variable_values.clone(),
            is_feasible: self.is_feasible(),
        }
    }
}

/// The population with the solutions.
#[derive(Clone, Default)]
pub struct Population(pub Vec<Individual>);

impl Population {
    /// Initialise a population with no individuals.
    ///
    /// returns: `Self`
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialise a population with some individuals.
    ///
    /// # Arguments
    ///
    /// * `individual`: The vector of individuals to add.
    ///
    /// returns: `Self`
    pub fn new_with(individuals: Vec<Individual>) -> Self {
        Self(individuals)
    }

    /// Get the population size.
    ///
    /// return: `usize`
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return `true` if the population is empty.
    ///
    /// return: `bool`
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the population individuals.
    ///
    /// return: `&[Individual]`
    pub fn individuals(&self) -> &[Individual] {
        self.0.as_ref()
    }

    /// Get a population individual by its index.
    ///
    /// return: `Option<&Individual>`
    pub fn individual(&self, index: usize) -> Option<&Individual> {
        self.0.get(index)
    }

    /// Borrow the population individuals as mutable reference.
    ///
    /// return: `&mut [Individual]`
    pub fn individuals_as_mut(&mut self) -> &mut [Individual] {
        self.0.as_mut()
    }

    /// Add new individuals to the population.
    ///
    /// # Arguments
    ///
    /// * `individuals`: The vector of individuals to add.
    ///
    /// returns: `()`
    pub fn add_new_individuals(&mut self, individuals: Vec<Individual>) {
        self.0.extend(individuals);
    }

    /// Add a new individual to the population.
    ///
    /// # Arguments
    ///
    /// * `individual`: The individual to add.
    ///
    /// returns: `()`
    pub fn add_individual(&mut self, individual: Individual) {
        self.0.push(individual);
    }

    /// Remove the specified range from the population in bulk and return all removed elements.
    ///
    /// # Arguments
    ///
    /// * `range_to_remove`: The range to remove.
    ///
    /// returns: `Vec<Individual>`
    pub fn drain<R>(&mut self, range_to_remove: R) -> Vec<Individual>
    where
        R: RangeBounds<usize>,
    {
        self.0.drain(range_to_remove).collect()
    }

    /// Generate a population with a number of individuals equal to `number_of_individuals`. All
    /// variable values for all individuals are initialised to an initial value depending on the
    /// variable type (for example min for a bounded real variable).  
    ///
    /// # Arguments
    ///
    /// * `problem`: The problem being solved.
    /// * `number_of_individuals`: The number of individuals to add to the population.
    ///
    /// returns: `Population`
    pub fn init(problem: Arc<Problem>, number_of_individuals: usize) -> Self {
        let mut population: Vec<Individual> = vec![];
        for _ in 0..number_of_individuals {
            population.push(Individual::new(problem.clone()));
        }
        Self(population)
    }

    /// Serialise the individuals for export.
    ///
    /// return: `Vec<IndividualExport>`
    pub fn serialise(&self) -> Vec<IndividualExport> {
        self.0.iter().map(|i| i.serialise()).collect()
    }
}

pub trait Individuals {
    fn individual(&self, index: usize) -> Result<&Individual, OError>;
    fn objective_values(&self, name: &str) -> Result<Vec<f64>, OError>;
    fn to_real_vec(&self, name: &str) -> Result<Vec<f64>, OError>;
}

pub trait IndividualsMut {
    fn individual_as_mut(&mut self, index: usize) -> Result<&mut Individual, OError>;
}

macro_rules! impl_individuals {
    ( $($type:ty),* $(,)? ) => {
        $(
            impl Individuals for $type {
                /// Get an individual from a vector.
                ///
                /// # Arguments
                ///
                /// * `index`: The index of the individual.
                ///
                /// return: `Result<&Individual, OError>`
                fn individual(&self, index: usize) -> Result<&Individual, OError> {
                    self.get(index)
                        .ok_or(OError::NonExistingIndex("individual".to_string(), index))
                }

                /// Get the objective values for all individuals. This returns an error if the objective name
                /// does not exist.
                ///
                /// # Arguments
                ///
                /// * `name`: The objective name.
                ///
                /// returns: `Result<f64, OError>`
                fn objective_values(&self, name: &str) -> Result<Vec<f64>, OError> {
                    self.iter().map(|i| i.get_objective_value(name)).collect()
                }

                /// Get the numbers stored in a real variable in all individuals. This returns an error if the
                /// variable does not exist or is not a real type.
                ///
                /// # Arguments
                ///
                /// * `name`: The variable name.
                ///
                /// returns: `Result<f64, OError>`
                fn to_real_vec(&self, name: &str) -> Result<Vec<f64>, OError> {
                    self.iter().map(|i| i.get_real_value(name)).collect()
                }
            }
        )*
    };
}

impl IndividualsMut for &mut [Individual] {
    /// Get a population individual as mutable.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the individual.
    ///
    /// return: `Result<&mut Individual, OError>`
    fn individual_as_mut(&mut self, index: usize) -> Result<&mut Individual, OError> {
        self.get_mut(index)
            .ok_or(OError::NonExistingIndex("individual".to_string(), index))
    }
}

// Implement methods for individuals for different types
impl_individuals!(&[Individual]);
impl_individuals!(&mut [Individual]);
impl_individuals!(Vec<Individual>);

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::core::{
        BoundedNumber, Constraint, Individual, Objective, ObjectiveDirection, Problem,
        RelationalOperator, VariableType,
    };
    use crate::core::utils::dummy_evaluator;

    #[test]
    /// Test when an objective does not exist
    fn test_non_existing_data() {
        let objectives = vec![Objective::new("objX", ObjectiveDirection::Minimise)];
        let var_types = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let e = dummy_evaluator();

        let problem = Arc::new(Problem::new(objectives, var_types, None, e).unwrap());
        let mut solution1 = Individual::new(problem);

        assert!(solution1.update_objective("obj1", 5.0).is_err());
        assert!(solution1.get_objective_value("obj1").is_err());
    }

    #[test]
    /// The is_feasible and constraint violation
    fn test_feasibility() {
        let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let constraints = vec![
            Constraint::new("c1", RelationalOperator::EqualTo, 1.0),
            Constraint::new("c2", RelationalOperator::EqualTo, 599.0),
        ];
        let e = dummy_evaluator();
        let problem = Arc::new(Problem::new(objectives, variables, Some(constraints), e).unwrap());

        let mut solution1 = Individual::new(problem);
        solution1.update_objective("obj1", 5.0).unwrap();

        // Unfeasible solution
        solution1.update_constraint("c1", 5.0).unwrap();
        assert!(!solution1.is_feasible());

        // Feasible solution
        solution1.update_constraint("c1", 1.0).unwrap();
        solution1.update_constraint("c2", 599.0).unwrap();
        assert!(solution1.is_feasible());

        // Total violation
        solution1.update_constraint("c1", 2.0).unwrap();
        solution1.update_constraint("c2", 600.0).unwrap();
        assert_eq!(solution1.constraint_violation(), 2.0);
    }
}
