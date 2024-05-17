use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

use log::info;
use serde::Serialize;

use crate::core::{Constraint, Individual, Objective, ObjectiveDirection, OError, VariableType};

/// The struct containing the results of the evaluation function. This is the output of
/// [`Evaluator::evaluate`], the user-defined function should produce.
#[derive(Debug)]
pub struct EvaluationResult {
    pub constraints: HashMap<String, f64>,
    pub objectives: HashMap<String, f64>,
}

/// The trait to use to evaluate the objective and constraint values when a new offspring is
/// created.
pub trait Evaluator: Sync + Send + Debug {
    /// A custom-defined function to use to assess the constraint and objective. When a new
    /// offspring is generated via crossover and mutation, new variables (or solutions) are
    /// assigned to it and the problem constraints and objective need to be evaluated. This
    /// function must return all the values for all the objectives and constraints set on
    /// the problem. An algorithm will return an error if the function fails to do so.
    ///
    /// # Arguments
    ///
    /// * `individual`: The individual.
    ///
    /// returns: `Result<EvaluationResult, Box<dyn Error>>`
    ///
    /// ## Example
    /// ```
    /// // access new variables to evaluate for the new individuals
    /// use std::error::Error;
    /// use optirustic::core::{EvaluationResult, Individual};
    /// fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
    ///     todo!()
    ///
    /// }
    /// ```
    fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>>;
}

#[derive(Serialize)]
pub struct ProblemExport {
    /// The problem objectives.
    objectives: HashMap<String, Objective>,
    /// The problem constraints.
    constraints: HashMap<String, Constraint>,
    /// The problem variables.
    variables: HashMap<String, VariableType>,
    /// The constraint names.
    constraint_names: Vec<String>,
    /// The variable names.
    variable_names: Vec<String>,
    /// The objective names.
    objective_names: Vec<String>,
    /// The number of objectives
    number_of_objectives: usize,
    /// The number of constraints
    number_of_constraints: usize,
    /// The number of variables
    number_of_variables: usize,
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
///  use std::error::Error;
///  use optirustic::core::{BoundedNumber, Constraint, EvaluationResult, Evaluator, Individual, Objective, ObjectiveDirection, Problem, RelationalOperator, VariableType};
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
///  #[derive(Debug)]
///  struct UserEvaluator;
///  impl Evaluator for UserEvaluator {
///     fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
///         Ok(EvaluationResult {
///             constraints: Default::default(),
///             objectives: Default::default(),
///         })
///     }
///  }
///
///  let problem = Problem::new(objectives, variables, Some(constraints), Box::new(UserEvaluator{})).unwrap();
///  println!("{}", problem);
/// ```
#[derive(Debug)]
pub struct Problem {
    /// The problem objectives.
    objectives: HashMap<String, Objective>,
    /// The problem constraints.
    constraints: HashMap<String, Constraint>,
    /// The problem variable types.
    variables: HashMap<String, VariableType>,
    /// The trait with the function to use to evaluate the objective and constraint values of
    /// new offsprings.
    evaluator: Box<dyn Evaluator>,
}

impl Display for Problem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Problem with {} variables, {} objectives and {} constraints",
            self.number_of_variables(),
            self.number_of_objectives(),
            self.number_of_constraints(),
        )
    }
}
impl Problem {
    /// Initialise the problem.
    ///
    /// # Arguments
    ///
    /// * `objectives`: The vector of objective to set on the problem.
    /// * `variable_types`: The vector of variable types to set on the problem.
    /// * `constraints`: The optional vector of constraints.
    /// * `evaluator`: The trait with the function to use to evaluate the objective and constraint
    /// values when new offsprings are generated by an algorithm. The [`Evaluator::evaluate`]
    /// receives the [`Individual`] with the new variables/solutions and should return the
    /// [`EvaluationResult`].
    ///
    /// returns: `Result<Problem, OError>`
    pub fn new(
        objectives: Vec<Objective>,
        variable_types: Vec<VariableType>,
        constraints: Option<Vec<Constraint>>,
        evaluator: Box<dyn Evaluator>,
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
            let name = objective.name();
            if p_objectives.contains_key(&name) {
                return Err(OError::DuplicatedName("objective".to_string(), name));
            }
            info!("Adding objective '{}' - {}", name, objective);
            p_objectives.insert(name, objective);
        }

        let mut p_variables: HashMap<String, VariableType> = HashMap::new();
        for var_type in variable_types.into_iter() {
            let name = var_type.name().to_string();
            if p_variables.contains_key(&name) {
                return Err(OError::DuplicatedName("variable".to_string(), name));
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
                        return Err(OError::DuplicatedName("constraint".to_string(), name));
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
            evaluator,
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
        Ok(self.objectives[name].direction() == ObjectiveDirection::Minimise)
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
    /// return `HashMap<String, VariableType>`
    pub fn variables(&self) -> HashMap<String, VariableType> {
        self.variables.clone()
    }

    /// Get a variable type by name. This returns an error if the variable does not exist.
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
    /// return `HashMap<String, Constraint>`
    pub fn constraints(&self) -> HashMap<String, Constraint> {
        self.constraints.clone()
    }

    /// The function used to evaluate the constraint and objective values for a new offsprings.
    ///
    /// return `&Evaluator`
    pub fn evaluator(&self) -> &dyn Evaluator {
        self.evaluator.as_ref()
    }

    /// Serialise the problem data.
    ///
    /// return: `ProblemExport`
    pub fn serialise(&self) -> ProblemExport {
        ProblemExport {
            objectives: self.objectives.clone(),
            constraints: self.constraints.clone(),
            variables: self.variables.clone(),
            constraint_names: self.constraint_names().clone(),
            variable_names: self.variable_names(),
            objective_names: self.objective_names().clone(),
            number_of_objectives: self.number_of_objectives(),
            number_of_constraints: self.number_of_constraints(),
            number_of_variables: self.number_of_variables(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::core::{
        BoundedNumber, Constraint, Objective, ObjectiveDirection, Problem, RelationalOperator,
        VariableType,
    };
    use crate::core::utils::dummy_evaluator;

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
        let e = dummy_evaluator();

        assert!(Problem::new(objectives, var_types, None, e).is_err());

        let e = dummy_evaluator();
        let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
        let constraints = vec![
            Constraint::new("c1", RelationalOperator::EqualTo, 1.0),
            Constraint::new("c1", RelationalOperator::GreaterThan, -1.0),
        ];
        assert!(Problem::new(objectives, var_types2, Some(constraints), e).is_err());
    }
}
