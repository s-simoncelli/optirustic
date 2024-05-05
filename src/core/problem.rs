use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use log::info;
use serde::Serialize;

use crate::core::{Constraint, VariableType};

/// Whether the objective should be minimised or maximised. Default is minimise.
#[derive(Default, Debug, PartialOrd, PartialEq, Serialize)]
pub enum ObjectiveDirection {
    #[default]
    Minimise,
    Maximise,
}

impl Display for ObjectiveDirection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectiveDirection::Minimise => f.write_str("minimised"),
            ObjectiveDirection::Maximise => f.write_str("maximised"),
        }
    }
}

#[derive(Serialize, Debug)]
pub struct Objective {
    /// The objective name.
    name: String,
    /// Whether the objective should be minimised or maximised.
    direction: ObjectiveDirection,
}

impl Objective {
    /// Create a new objective.
    ///
    /// # Arguments
    ///
    /// * `name`: The objective name.
    /// * `direction`:  Whether the objective should be minimised or maximised.
    ///
    /// returns: `Objective`
    pub fn new(name: &str, direction: ObjectiveDirection) -> Self {
        Self {
            name: name.to_string(),
            direction,
        }
    }

    /// Get the constraint name.
    pub fn name(&self) -> String {
        self.name.clone()
    }
}

impl Display for Objective {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Objective '{}' is {}", self.name, self.direction)
    }
}

/// Define a new problem to optimise.
#[derive(Default, Debug)]
pub struct Problem {
    /// The problem objectives.
    objectives: HashMap<String, Objective>,
    /// The problem constraints.
    constraints: HashMap<String, Constraint>,
    /// The problem variable types.
    variables: HashMap<String, VariableType>,
}

#[derive(Serialize)]
pub struct ProblemExport {
    /// The problem objectives.
    objectives: HashMap<String, Objective>,
    /// The objective names.
    objective_names: Vec<String>,
    /// The problem constraints.
    constraints: HashMap<String, Constraint>,
    /// The constraint names.
    constraint_names: Vec<String>,
    //The number of objectives
    number_of_objectives: usize,
    //The number of constraints
    number_of_constraints: usize,
}

impl Problem {
    /// Initialise the problem.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            objectives: HashMap::new(),
            constraints: HashMap::new(),
        }
    }

    /// Whether a problem objective is being minimised.
    ///
    /// # Arguments
    ///
    /// * `name`: The objective name.
    ///
    ///
    /// returns: `Result<bool, String>`
    pub fn is_objective_minimised(&self, name: &str) -> Result<bool, String> {
        if !self.objectives.contains_key(name) {
            return Err(format!("The objective named '{}' does not exist", name));
        }
        Ok(self.objectives[name].direction == ObjectiveDirection::Minimise)
    }

    /// Get the total number of objectives of the problem.
    ///
    /// returns: `usize`
    pub fn number_of_objectives(&self) -> usize {
        self.objectives.len()
    }

    /// Get the total number of constraints of the problem.
    ///
    /// returns: `usize`
    pub fn number_of_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Get the total number of variables of the problem.
    ///
    /// returns: `usize`
    pub fn number_of_variables(&self) -> usize {
        self.variables.len()
    }

    /// Add a new variable type to the problem.
    ///
    /// # Arguments
    ///
    /// * `name`: The variable name.
    /// * `variable_type`: The variable type to add.
    ///
    /// returns: `Result<(), String>` An error is returned if an variable with the same name
    /// already exists.
    pub fn add_variable(&mut self, name: &str, variable_type: VariableType) -> Result<(), String> {
        if self.variables.contains_key(name) {
            return Err(format!("The variable type named '{}' already exist", name));
        }
        info!("Adding variable '{}' - {}", name, variable_type);
        self.variables.insert(name.to_string(), variable_type);
        Ok(())
    }

    /// Add a new objective to the problem.
    ///
    /// # Arguments
    ///
    /// * `objective`: The objective to add.
    ///
    /// returns: `Result<(), String>` An error is returned if an objective with the same name
    /// already exists.
    pub fn add_objective(&mut self, objective: Objective) -> Result<(), String> {
        let name = objective.name();
        if self.objectives.contains_key(&name) {
            return Err(format!("The objective named '{}' already exist", name));
        }
        info!("Adding objective '{}' - {}", name, objective);
        self.objectives.insert(name, objective);
        Ok(())
    }

    /// Add a new constraint to the problem.
    ///
    /// # Arguments
    ///
    /// * `constraint`: The constraint to add.
    ///
    /// returns: `Result<(), String>` An error is returned if a constraint with the same name
    /// already exists.
    pub fn add_constraint(&mut self, constraint: Constraint) -> Result<(), String> {
        let name = constraint.name();
        if self.constraints.contains_key(&name) {
            return Err(format!("The constraint named '{}' already exist", name));
        }
        info!("Adding constraint '{}' - {}", name, constraint);
        self.constraints.insert(name, constraint);
        Ok(())
    }

    /// Get the name of the variables set on the problem.
    ///
    /// return `Vec<String>`
    pub fn variable_names(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }

    /// Get the name of the objectives set on the problem.
    ///
    /// return `Vec<String>`
    pub fn objective_names(&self) -> Vec<String> {
        self.objectives.keys().cloned().collect()
    }

    /// Get the name of the constraints set on the problem.
    ///
    /// return `Vec<String>`
    pub fn constraint_names(&self) -> Vec<String> {
        self.constraints.keys().cloned().collect()
    }

    /// Get the map of variables.
    ///
    /// return `HashMap<VariableType>`
    pub fn variables(&self) -> HashMap<String, VariableType> {
        self.variables.clone()
    }

    /// Get a variable type by name.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the variable to fetch.
    ///
    /// return `Result<VariableType, String>`
    pub fn get_variable(&self, name: &str) -> Result<VariableType, String> {
        if !self.variable_names().contains(&name.to_string()) {
            return Err(format!("The variable named '{}' does not exist", name));
        }
        Ok(self.variables[name].clone())
    }

    /// Get the list of constraints.
    ///
    /// return `HashMap<Constraint>`
    pub fn constraints(&self) -> HashMap<String, Constraint> {
        self.constraints.clone()
    }

    /// Export the problem data.
    ///
    /// return: `ProblemExport`
    pub fn export(&self) {
        //     let output = ProblemExport {
        //         objectives: self.objectives.iter().clone().collect(),
        //         constraints: self.constraints.clone(),
        //         objective_names: self.objective_names().clone(),
        //         constraint_names: self.constraint_names().clone(),
        //         number_of_objectives: self.number_of_objectives(),
        //         number_of_constraints: self.number_of_constraints(),
        //     };
        //     Ok(serde_json::to_string_pretty(&output)?)
    }
}

#[cfg(test)]
mod test {
    use crate::core::{Constraint, Objective, ObjectiveDirection, Problem, RelationalOperator};

    #[test]
    /// Test when objectives and constraints already exist
    fn test_already_existing_data() {
        let mut problem = Problem::new();
        problem
            .add_objective(Objective::new("obj1", ObjectiveDirection::Minimise))
            .unwrap();
        assert!(problem
            .add_objective(Objective::new("obj1", ObjectiveDirection::Maximise))
            .is_err());

        problem
            .add_constraint(Constraint::new("c1", RelationalOperator::EqualTo, 1.0))
            .unwrap();
        assert!(problem
            .add_constraint(Constraint::new("c1", RelationalOperator::GreaterThan, -1.0))
            .is_err());
    }
}
