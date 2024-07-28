use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::core::{Constraint, Individual, Objective, ObjectiveDirection, OError, VariableType};

/// The struct containing the results of the evaluation function. This is the output of
/// [`Evaluator::evaluate`], the user-defined function should produce. When the algorithm generates
/// a new population with new variables, its constraints and objectives must be evaluated to proceed
/// with the next evolution.
#[derive(Debug)]
pub struct EvaluationResult {
    /// The list of evaluated constraints. This is optional for unconstrained problems.
    pub constraints: Option<HashMap<String, f64>>,
    /// The list of evaluated objectives.
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
    /// use std::collections::HashMap;
    /// use std::error::Error;
    /// use optirustic::core::{EvaluationResult, Individual, Evaluator};
    ///
    /// // solve a SCH problem with two objectives to minimise: x^2 and (x-2)^2. The problem has
    /// // one variables named "x" and two objectives named "x^2" and "(x-2)^2".
    /// #[derive(Debug)]
    ///     struct UserEvaluator;
    ///     impl Evaluator for UserEvaluator {
    ///         fn evaluate(&self, i: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
    ///             // access new variable to evaluate the objectives
    ///             let x = i.get_variable_value("x")?.as_real()?;
    ///             // assess the objectives
    ///             let mut objectives = HashMap::new();
    ///             objectives.insert("x^2".to_string(), x.powi(2));
    ///             objectives.insert("(x-2)^2".to_string(), (x - 2.0).powi(2));
    ///
    ///             Ok(EvaluationResult {
    ///                 constraints: None,
    ///                 objectives,
    ///             })
    ///         }
    ///     }
    /// ```
    fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>>;
}

#[derive(Serialize, Deserialize, Debug)]
/// Serialised data of a problem.
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
    objectives: Vec<Objective>,
    /// The problem constraints.
    constraints: Vec<Constraint>,
    /// The problem variable types.
    variables: Vec<VariableType>,
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
        let constraints = constraints.unwrap_or_default();

        Ok(Self {
            variables: variable_types,
            objectives,
            constraints,
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
        match self.objectives.iter().position(|o| o.name() == name) {
            None => Err(OError::NonExistingName(
                "objective".to_string(),
                name.to_string(),
            )),
            Some(p) => Ok(self.objectives[p].direction() == ObjectiveDirection::Minimise),
        }
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
        self.variables.iter().map(|o| o.name()).collect()
    }

    /// Get the name of the objectives set on the problem.
    ///
    /// return `Vec<String>`
    pub fn objective_names(&self) -> Vec<String> {
        self.objectives.iter().map(|o| o.name()).collect()
    }

    /// Get the name of the constraints set on the problem.
    ///
    /// return `Vec<String>`
    pub fn constraint_names(&self) -> Vec<String> {
        self.constraints.iter().map(|o| o.name()).collect()
    }

    /// Get the map of variables.
    ///
    /// return `Vec<(String, VariableType)>`
    pub fn variables(&self) -> Vec<(String, VariableType)> {
        self.variables
            .iter()
            .map(|o| (o.name().clone(), o.clone()))
            .collect()
    }

    /// Get a variable type by name. This returns an error if the variable does not exist.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the variable to fetch.
    ///
    /// return `Result<VariableType, OError>`
    pub fn get_variable(&self, name: &str) -> Result<VariableType, OError> {
        match self.variables.iter().position(|o| o.name() == name) {
            None => Err(OError::NonExistingName(
                "variable".to_string(),
                name.to_string(),
            )),
            Some(p) => Ok(self.variables[p].clone()),
        }
    }

    /// Get a constraint by name. This returns an error if the constraint does not exist.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the constraint to fetch.
    ///
    /// return `Result<Constraint, OError>`
    pub fn get_constraint(&self, name: &str) -> Result<Constraint, OError> {
        match self.constraints.iter().position(|o| o.name() == name) {
            None => Err(OError::NonExistingName(
                "variable".to_string(),
                name.to_string(),
            )),
            Some(p) => Ok(self.constraints[p].clone()),
        }
    }

    /// Get the list of objectives.
    ///
    /// return `Vec<(String, Objective)>`
    pub fn objectives(&self) -> Vec<(String, Objective)> {
        self.objectives
            .iter()
            .map(|o| (o.name().clone(), o.clone()))
            .collect()
    }

    /// Get the list of constraints.
    ///
    /// return `Vec<(String, Constraint)>`
    pub fn constraints(&self) -> Vec<(String, Constraint)> {
        self.constraints
            .iter()
            .map(|o| (o.name().clone(), o.clone()))
            .collect()
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
        let objectives: HashMap<String, Objective> = self
            .objectives()
            .iter()
            .map(|(name, obj)| (name.clone(), obj.clone()))
            .collect();
        let constraints: HashMap<String, Constraint> = self
            .constraints()
            .iter()
            .map(|(name, c)| (name.clone(), c.clone()))
            .collect();
        let variables: HashMap<String, VariableType> = self
            .variables()
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();

        ProblemExport {
            objectives,
            constraints,
            variables,
            constraint_names: self.constraint_names().clone(),
            variable_names: self.variable_names(),
            objective_names: self.objective_names().clone(),
            number_of_objectives: self.number_of_objectives(),
            number_of_constraints: self.number_of_constraints(),
            number_of_variables: self.number_of_variables(),
        }
    }
}

pub mod builtin_problems {
    use std::collections::HashMap;
    use std::error::Error;
    use std::f64::consts::PI;

    use crate::core::{
        BoundedNumber, EvaluationResult, Evaluator, Individual, Objective, ObjectiveDirection,
        OError, Problem, VariableType,
    };

    /// The Schaffer’s study (SCH) problem.
    #[derive(Debug)]
    pub struct SCHProblem;

    impl SCHProblem {
        /// Create the problem for the optimisation.
        pub fn create() -> Result<Problem, OError> {
            let objectives = vec![
                Objective::new("x^2", ObjectiveDirection::Minimise),
                Objective::new("(x-2)^2", ObjectiveDirection::Minimise),
            ];
            let variables = vec![VariableType::Real(BoundedNumber::new(
                "x", -1000.0, 1000.0,
            )?)];

            let e = Box::new(SCHProblem);
            Problem::new(objectives, variables, None, e)
        }

        /// The first objective function
        pub fn f1(x: f64) -> f64 {
            x.powi(2)
        }

        /// The second objective function
        pub fn f2(x: f64) -> f64 {
            (x - 2.0).powi(2)
        }
    }

    impl Evaluator for SCHProblem {
        fn evaluate(&self, i: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
            let x = i.get_variable_value("x")?.as_real()?;
            let mut objectives = HashMap::new();
            objectives.insert("x^2".to_string(), SCHProblem::f1(x));
            objectives.insert("(x-2)^2".to_string(), SCHProblem::f2(x));
            Ok(EvaluationResult {
                constraints: None,
                objectives,
            })
        }
    }

    /// The Fonseca and Fleming’s study (FON) problem.
    #[derive(Debug)]
    pub struct FonProblem;

    impl FonProblem {
        /// Create the problem for the optimisation.
        pub fn create() -> Result<Problem, OError> {
            let objectives = vec![
                Objective::new("f1", ObjectiveDirection::Minimise),
                Objective::new("f2", ObjectiveDirection::Minimise),
            ];
            let variables = vec![
                VariableType::Real(BoundedNumber::new("x1", -4.0, 4.0)?),
                VariableType::Real(BoundedNumber::new("x2", -4.0, 4.0)?),
                VariableType::Real(BoundedNumber::new("x3", -4.0, 4.0)?),
            ];

            let e = Box::new(FonProblem);
            Problem::new(objectives, variables, None, e)
        }
    }

    impl Evaluator for FonProblem {
        fn evaluate(&self, i: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
            let mut x: Vec<f64> = Vec::new();
            for var_name in ["x1", "x2", "x3"] {
                x.push(i.get_variable_value(var_name)?.as_real()?);
            }
            let mut objectives = HashMap::new();

            let mut exp_arg1 = 0.0;
            let mut exp_arg2 = 0.0;
            for x_val in x {
                exp_arg1 += (x_val - 1.0 / 3.0_f64.sqrt()).powi(2);
                exp_arg2 += (x_val + 1.0 / 3.0_f64.sqrt()).powi(2);
            }
            objectives.insert("f1".to_string(), 1.0 - f64::exp(-exp_arg1));
            objectives.insert("f2".to_string(), 1.0 - f64::exp(-exp_arg2));
            Ok(EvaluationResult {
                constraints: None,
                objectives,
            })
        }
    }

    /// Problem #1 from Zitzler et al. (2000).
    #[derive(Debug)]
    pub struct ZTD1Problem {
        /// The number of variables.
        n: usize,
    }

    impl ZTD1Problem {
        /// Create the problem for the optimisation.
        ///
        /// # Arguments:
        ///
        /// * `n`: The number of variables.
        pub fn create(n: usize) -> Result<Problem, OError> {
            let objectives = vec![
                Objective::new("f1", ObjectiveDirection::Minimise),
                Objective::new("f2", ObjectiveDirection::Minimise),
            ];
            let mut variables: Vec<VariableType> = Vec::new();
            for i in 1..=n {
                variables.push(VariableType::Real(BoundedNumber::new(
                    format!("x{i}").as_str(),
                    0.0,
                    1.0,
                )?));
            }

            let e = Box::new(ZTD1Problem { n });
            Problem::new(objectives, variables, None, e)
        }

        /// The first objective function.
        pub fn f1(x: &[f64]) -> f64 {
            x[0]
        }

        /// The second objective function.
        pub fn f2(&self, x: &[f64]) -> f64 {
            let a: f64 = (1..self.n).map(|xi| x[xi]).sum();
            let g = 1.0 + 9.0 * a / (self.n as f64 - 1.0);
            g * (1.0 - f64::sqrt(x[0] / g))
        }
    }
    impl Evaluator for ZTD1Problem {
        fn evaluate(&self, i: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
            let x: Vec<f64> = i
                .get_variable_values()?
                .iter()
                .map(|v| v.as_real())
                .collect::<Result<Vec<f64>, _>>()?;

            let mut objectives = HashMap::new();
            objectives.insert("f1".to_string(), ZTD1Problem::f1(&x));
            objectives.insert("f2".to_string(), self.f2(&x));
            Ok(EvaluationResult {
                constraints: None,
                objectives,
            })
        }
    }

    /// Problem #2 from Zitzler et al. (2000)
    #[derive(Debug)]
    pub struct ZTD2Problem {
        /// The number of variables.
        n: usize,
    }

    impl ZTD2Problem {
        /// Create the problem for the optimisation.
        ///
        /// # Arguments:
        ///
        /// * `n`: The number of variables.
        pub fn create(n: usize) -> Result<Problem, OError> {
            let objectives = vec![
                Objective::new("f1", ObjectiveDirection::Minimise),
                Objective::new("f2", ObjectiveDirection::Minimise),
            ];
            let mut variables: Vec<VariableType> = Vec::new();
            for i in 1..=n {
                variables.push(VariableType::Real(BoundedNumber::new(
                    format!("x{i}").as_str(),
                    0.0,
                    1.0,
                )?));
            }

            let e = Box::new(ZTD2Problem { n });
            Problem::new(objectives, variables, None, e)
        }

        /// The first objective function.
        pub fn f1(x: &[f64]) -> f64 {
            x[0]
        }

        /// The second objective function.
        pub fn f2(&self, x: &[f64]) -> f64 {
            let a: f64 = (1..self.n).map(|xi| x[xi]).sum();
            let g = 1.0 + 9.0 * a / (self.n as f64 - 1.0);
            g * (1.0 - (x[0] / g).powi(2))
        }
    }

    impl Evaluator for ZTD2Problem {
        fn evaluate(&self, i: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
            let x: Vec<f64> = i
                .get_variable_values()?
                .iter()
                .map(|v| v.as_real())
                .collect::<Result<Vec<f64>, _>>()?;

            let mut objectives = HashMap::new();
            objectives.insert("f1".to_string(), ZTD2Problem::f1(&x));
            objectives.insert("f2".to_string(), self.f2(&x));
            Ok(EvaluationResult {
                constraints: None,
                objectives,
            })
        }
    }

    /// Problem #3 from Zitzler et al. (2000)
    #[derive(Debug)]
    pub struct ZTD3Problem {
        /// The number of variables.
        n: usize,
    }

    impl ZTD3Problem {
        /// Create the problem for the optimisation.
        ///
        /// # Arguments:
        ///
        /// * `n`: The number of variables.
        pub fn create(n: usize) -> Result<Problem, OError> {
            let objectives = vec![
                Objective::new("f1", ObjectiveDirection::Minimise),
                Objective::new("f2", ObjectiveDirection::Minimise),
            ];
            let mut variables: Vec<VariableType> = Vec::new();
            for i in 1..=n {
                variables.push(VariableType::Real(BoundedNumber::new(
                    format!("x{i}").as_str(),
                    0.0,
                    1.0,
                )?));
            }

            let e = Box::new(ZTD3Problem { n });
            Problem::new(objectives, variables, None, e)
        }

        /// The first objective function.
        pub fn f1(x: &[f64]) -> f64 {
            x[0]
        }

        /// The second objective function.
        pub fn f2(&self, x: &[f64]) -> f64 {
            let a: f64 = (1..self.n).map(|xi| x[xi]).sum();
            let g = 1.0 + 9.0 * a / (self.n as f64 - 1.0);
            g * (1.0 - (x[0] / g).powi(2) - x[0] / g * f64::sin(10.0 * PI * x[0]))
        }
    }

    impl Evaluator for ZTD3Problem {
        fn evaluate(&self, i: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
            let x: Vec<f64> = i
                .get_variable_values()?
                .iter()
                .map(|v| v.as_real())
                .collect::<Result<Vec<f64>, _>>()?;

            let mut objectives = HashMap::new();
            objectives.insert("f1".to_string(), ZTD3Problem::f1(&x));
            objectives.insert("f2".to_string(), self.f2(&x));
            Ok(EvaluationResult {
                constraints: None,
                objectives,
            })
        }
    }

    /// Problem #4 from Zitzler et al. (2000)
    #[derive(Debug)]
    pub struct ZTD4Problem {
        /// The number of variables.
        n: usize,
    }

    impl ZTD4Problem {
        /// Create the problem for the optimisation.
        ///
        /// # Arguments:
        ///
        /// * `n`: The number of variables.
        pub fn create(n: usize) -> Result<Problem, OError> {
            let objectives = vec![
                Objective::new("f1", ObjectiveDirection::Minimise),
                Objective::new("f2", ObjectiveDirection::Minimise),
            ];
            let mut variables: Vec<VariableType> = Vec::new();
            variables.push(VariableType::Real(BoundedNumber::new("x1", 0.0, 1.0)?));
            for i in 2..=n {
                variables.push(VariableType::Real(BoundedNumber::new(
                    format!("x{i}").as_str(),
                    -5.0,
                    5.0,
                )?));
            }
            let e = Box::new(ZTD4Problem { n });
            Problem::new(objectives, variables, None, e)
        }

        /// The first objective function.
        pub fn f1(x: &[f64]) -> f64 {
            x[0]
        }

        /// The second objective function.
        pub fn f2(&self, x: &[f64]) -> f64 {
            let a: f64 = (1..self.n)
                .map(|xi| {
                    let xi = x[xi];
                    xi.powi(2) - 10.0 * f64::cos(4.0 * PI * xi)
                })
                .sum();
            let g: f64 = 1.0 + 10.0 * (self.n as f64 - 1.0) + a;

            g * (1.0 - (x[0] / g).sqrt())
        }
    }

    impl Evaluator for ZTD4Problem {
        fn evaluate(&self, i: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
            let x: Vec<f64> = i
                .get_variable_values()?
                .iter()
                .map(|v| v.as_real())
                .collect::<Result<Vec<f64>, _>>()?;

            let mut objectives = HashMap::new();
            objectives.insert("f1".to_string(), ZTD4Problem::f1(&x));
            objectives.insert("f2".to_string(), self.f2(&x));
            Ok(EvaluationResult {
                constraints: None,
                objectives,
            })
        }
    }

    /// Problem #6 from Zitzler et al. (2000)
    #[derive(Debug)]
    pub struct ZTD6Problem {
        /// The number of variables.
        n: usize,
    }

    impl ZTD6Problem {
        /// Create the problem for the optimisation.
        ///
        /// # Arguments:
        ///
        /// * `n`: The number of variables.
        pub fn create(n: usize) -> Result<Problem, OError> {
            let objectives = vec![
                Objective::new("f1", ObjectiveDirection::Minimise),
                Objective::new("f2", ObjectiveDirection::Minimise),
            ];
            let mut variables: Vec<VariableType> = Vec::new();
            for i in 1..=n {
                variables.push(VariableType::Real(BoundedNumber::new(
                    format!("x{i}").as_str(),
                    0.0,
                    1.0,
                )?));
            }

            let e = Box::new(ZTD6Problem { n });
            Problem::new(objectives, variables, None, e)
        }

        /// The first objective function.
        pub fn f1(x: &[f64]) -> f64 {
            1.0 - f64::exp(-4.0 * x[0]) * f64::powi(f64::sin(6.0 * PI * x[0]), 6)
        }

        /// The second objective function.
        pub fn f2(&self, x: &[f64]) -> f64 {
            let a = (1..self.n).map(|xi| x[xi]).sum::<f64>() / (self.n as f64 - 1.0);
            let g = 1.0 + 9.0 * f64::powf(a, 0.25);
            g * (1.0 - (ZTD6Problem::f1(&x) / g).powi(2))
        }
    }

    impl Evaluator for ZTD6Problem {
        fn evaluate(&self, i: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
            let x: Vec<f64> = i
                .get_variable_values()?
                .iter()
                .map(|v| v.as_real())
                .collect::<Result<Vec<f64>, _>>()?;

            let mut objectives = HashMap::new();
            objectives.insert("f1".to_string(), ZTD6Problem::f1(&x));
            objectives.insert("f2".to_string(), self.f2(&x));
            Ok(EvaluationResult {
                constraints: None,
                objectives,
            })
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
