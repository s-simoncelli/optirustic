use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use log::info;
use serde::Serialize;

use crate::core::{Constraint, VariableType};
use crate::core::error::OError;
use crate::core::error::OError::DuplicatedName;

/// Whether the objective should be minimised or maximised. Default is minimise.
#[derive(Default, Debug, PartialOrd, PartialEq, Serialize)]
pub enum ObjectiveDirection {
    #[default]
    /// Minimise an objective.
    Minimise,
    /// Maximise an objective.
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

/// Define a problem objective to minimise or maximise.
///
/// # Example
/// ```
///  use optirustic::core::{Objective, ObjectiveDirection};
///
///  let o = Objective::new("Reduce cost", ObjectiveDirection::Minimise);
///  println!("{}", o);
/// ```
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

/// Define a new problem to optimise as:
///
///  $$$ Min/Max(f_1(x), f_2(x), ..., f_M(x)) $
///
/// where
///   - where the integer $M \geq 1$ is the number of objectives;
///   - $x$ the $N$-variable solution vector bounded to $$$ x_i^{(L)} \leq x_i \leq x_i^{(U)}$ with
/// $i=1,2,...,N$.
///
/// The problem is also subjected to the following constraints:
/// - $$$ g_j(x) \geq 0 $ with $j=1,2,...,J$ and $J$ the number of inequality constraints.
/// - $$$ h_k(x) = 0 $ with $k=1,2,...,H$ and $H$ the number of equality constraints.
///
/// # Example
/// ```
///  use optirustic::core::{BoundedNumber, Constraint, Objective, ObjectiveDirection, Problem, RelationalOperator, VariableType};
///
///  // Define a one-objective one-variable problem with two constraints
///  let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
///  let variables = vec![VariableType::Real(
///     BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
///  )];
///  let constraints = vec![
///     Constraint::new("c1", RelationalOperator::EqualTo, 1.0),
///     Constraint::new("c2", RelationalOperator::EqualTo, 599.0),
///  ];
///
///  let problem = Problem::new(objectives, variables, Some(constraints)).unwrap();
///  println!("{}", problem);
/// ```
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
    ///
    /// # Arguments
    ///
    /// * `objectives`: The vector of objective to set on the problem.
    /// * `variable_types`: The vector of variable types to set on the problem.
    /// * `constraints`: The optional vector of constraints.
    ///
    /// returns: `Result<Problem, OError>`
    pub fn new(
        objectives: Vec<Objective>,
        variable_types: Vec<VariableType>,
        constraints: Option<Vec<Constraint>>,
    ) -> Result<Self, OError> {
        // Check vector lengths
        if objectives.is_empty() {
            return Err(OError::NoObjective);
        }
        if variable_types.is_empty() {
            return Err(OError::NoVariables);
        }

        let mut p_objectives: HashMap<String, Objective> = HashMap::new();
        for objective in objectives.into_iter() {
            let name = objective.name.to_string();
            if p_objectives.contains_key(&name) {
                return Err(DuplicatedName("objective".to_string(), name));
            }
            info!("Adding objective '{}' - {}", name, objective);
            p_objectives.insert(name, objective);
        }

        let mut p_variables: HashMap<String, VariableType> = HashMap::new();
        for var_type in variable_types.into_iter() {
            let name = var_type.name().to_string();
            if p_variables.contains_key(&name) {
                return Err(DuplicatedName("variable".to_string(), name));
            }
            info!("Adding variable type '{}' - {}", name, var_type);
            p_variables.insert(name, var_type);
        }

        let p_constraints = match constraints {
            None => HashMap::<String, Constraint>::new(),
            Some(c) => {
                let mut p_constraints: HashMap<String, Constraint> = HashMap::new();
                for constraint in c.into_iter() {
                    let name = constraint.name().to_string();
                    if p_constraints.contains_key(&name) {
                        return Err(DuplicatedName("constraint".to_string(), name));
                    }
                    info!("Adding constraint '{}' - {}", name, constraint);
                    p_constraints.insert(name, constraint);
                }
                p_constraints
            }
        };

        Ok(Self {
            variables: p_variables,
            objectives: p_objectives,
            constraints: p_constraints,
        })
    }

    /// Whether a problem objective is being minimised. This returns an error if the objective does
    /// not exist.
    ///
    /// # Arguments
    ///
    /// * `name`: The objective name.
    ///
    ///
    /// returns: `Result<bool, OError>`
    pub fn is_objective_minimised(&self, name: &str) -> Result<bool, OError> {
        if !self.objectives.contains_key(name) {
            return Err(OError::NonExistingName(
                "objective".to_string(),
                name.to_string(),
            ));
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
    /// return `Result<VariableType, OError>`
    pub fn get_variable(&self, name: &str) -> Result<VariableType, OError> {
        if !self.variable_names().contains(&name.to_string()) {
            return Err(OError::NonExistingName(
                "variable".to_string(),
                name.to_string(),
            ));
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
    use crate::core::{
        BoundedNumber, Constraint, Objective, ObjectiveDirection, Problem, RelationalOperator,
        VariableType,
    };

    #[test]
    /// Test when objectives and constraints already exist
    fn test_already_existing_data() {
        let objectives = vec![
            Objective::new("obj1", ObjectiveDirection::Minimise),
            Objective::new("obj1", ObjectiveDirection::Maximise),
        ];
        let var_types = vec![VariableType::Real(
            BoundedNumber::new("X1", 0.0, 2.0).unwrap(),
        )];
        let var_types2 = var_types.clone();

        assert!(Problem::new(objectives, var_types, None).is_err());

        let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
        let constraints = vec![
            Constraint::new("c1", RelationalOperator::EqualTo, 1.0),
            Constraint::new("c1", RelationalOperator::GreaterThan, -1.0),
        ];
        assert!(Problem::new(objectives, var_types2, Some(constraints)).is_err());
    }
}
