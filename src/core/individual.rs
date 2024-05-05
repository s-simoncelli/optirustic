use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use serde::Serialize;

use crate::core::problem::Problem;
use crate::core::variable::VariableValue;

/// An individual in the population containing the problem solution.
#[derive(Debug)]
pub struct Individual<'a> {
    /// The problem being solved
    problem: &'a Problem,
    /// The value of the problem variables for the individual.
    variable_values: HashMap<String, VariableValue>,
    /// The value of the constraints.
    constraint_values: HashMap<String, f64>,
    /// The values of the objectives.
    objective_values: HashMap<String, f64>,
}

#[derive(Serialize)]
pub struct IndividualExport {
    /// The value of the constraints.
    constraint_values: HashMap<String, f64>,
    /// The values of the objectives.
    objective_values: HashMap<String, f64>,
    /// The overall amount of violation of the solution constraints.
    constraint_violation: f64,
    /// Whether the solution meets all the problem constraints.
    is_feasible: bool,
}

impl<'a> Display for Individual<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Individual(variables={:?}, objectives={:?},constraints={:?})",
            self.variable_values, self.objective_values, self.constraint_values,
        )
    }
}

impl<'a> Individual<'a> {
    /// Create a new individual. An individual contains the solution after an evolution.
    ///
    /// # Arguments
    ///
    /// * `problem`: The problem being solved.
    ///
    /// returns: `Individual`
    pub fn new(problem: &'a Problem) -> Self {
        let mut variable_values: HashMap<String, VariableValue> = HashMap::new();
        for (variable_name, var_type) in problem.variables() {
            variable_values.insert(variable_name, var_type.initial_value());
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
        }
    }

    /// Get the problem being solved with the individual.
    ///
    /// return `&Problem`
    pub fn problem(&self) -> &Problem {
        self.problem
    }

    /// Update the objective for a solution.
    ///
    /// # Arguments
    ///
    /// * `name`: The objective to update.
    /// * `value`: The value to set.
    ///
    /// returns: `Result<(), String>`
    pub fn update_objective(&mut self, name: &str, value: f64) -> Result<(), String> {
        if !self.objective_values.contains_key(name) {
            return Err(format!("The objective named '{}' does not exist", name));
        }
        if let Some(x) = self.objective_values.get_mut(name) {
            *x = value;
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
    /// returns: `Result<(), String>`
    pub fn update_constraint(&mut self, name: &str, value: f64) -> Result<(), String> {
        if !self.constraint_values.contains_key(name) {
            return Err(format!("The constrained named '{}' does not exist", name));
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
        // TODO this may panic is problem has no constraint with same name in hashmap
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
            if !self.problem.constraints().contains_key(name) {
                continue;
            }
            if !self
                .problem
                .constraints()
                .get(name)
                .unwrap()
                .is_met(*constraint_value)
            {
                return false;
            }
        }
        true
    }

    /// Ge the objective value by name.
    ///
    /// # Arguments
    ///
    /// * `name`: The objective name.
    ///
    /// returns: `f64`
    pub fn get_objective_value(&self, name: &str) -> Result<f64, String> {
        if !self.objective_values.contains_key(name) {
            return Err(format!("The objective named '{}' does not exist", name));
        }

        Ok(self.objective_values[name])
    }

    /// Export all the solution data (constraint and objective values, constraint violation and
    /// feasibility).
    ///
    /// return: `SolutionExport`
    pub fn export(&self) -> IndividualExport {
        IndividualExport {
            constraint_values: self.constraint_values.clone(),
            objective_values: self.objective_values.clone(),
            constraint_violation: self.constraint_violation(),
            is_feasible: self.is_feasible(),
        }
    }
}

/// The population
pub struct Population<'a>(pub Vec<&'a Individual<'a>>);

impl<'a> Population<'a> {
    /// Get the population individuals.
    ///
    /// return: `&[Individual]` .
    pub fn individuals(&self) -> Vec<&Individual> {
        self.0.clone()
    }

    /// Get the population size.
    ///
    /// return: `usize` The number of individuals.
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Whether the population has no individuals.
    ///
    /// return: `bool`
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Generate a population with a number of individuals equal to `number_of_individuals`. All
    /// variable values for all individuals are initialised to an initial value.  
    ///
    /// # Arguments
    ///
    /// * `problem`: The problem being solved.
    /// * `number_of_individuals`: The number of individuals to add to the population.
    ///
    /// returns: `Vec<Individual>`
    pub fn init(problem: &Problem, number_of_individuals: usize) -> Vec<Individual> {
        let mut population: Vec<Individual> = vec![];
        for _ in 0..number_of_individuals {
            population.push(Individual::new(problem));
        }
        population
    }
}

#[cfg(test)]
mod test {
    use crate::core::{
        Constraint, Individual, Objective, ObjectiveDirection, Problem, RelationalOperator,
    };

    #[test]
    /// Test when an objective does not exist
    fn test_already_existing_data() {
        let problem = Problem::new();
        let mut solution1 = Individual::new(&problem);

        assert!(solution1.update_objective("obj1", 5.0).is_err());
        assert!(solution1.get_objective_value("obj1").is_err());
    }

    #[test]
    /// The is_feasible and constraint violation
    fn test_feasibility() {
        let mut problem = Problem::new();
        problem
            .add_objective(Objective::new("obj1", ObjectiveDirection::Minimise))
            .unwrap();
        problem
            .add_constraint(Constraint::new("c1", RelationalOperator::EqualTo, 1.0))
            .unwrap();
        problem
            .add_constraint(Constraint::new("c2", RelationalOperator::EqualTo, 599.0))
            .unwrap();

        let mut solution1 = Individual::new(&problem);
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
