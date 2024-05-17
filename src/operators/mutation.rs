use rand::Rng;

use crate::core::{Individual, OError, Problem, VariableType, VariableValue};

/// The trait to implement a mutation operator to modify the genetic material of an individual.
pub trait Mutation {
    /// Mutate a population individual.
    ///
    /// # Arguments
    ///
    /// * `individual`: The individual to mutate.
    ///
    /// returns: `Result<Individual, OError>`. The mutated individual.
    fn mutate_offsprings(&self, individual: &Individual) -> Result<Individual, OError>;
}

/// Input arguments for [`PolynomialMutation`].
#[derive(Clone)]
pub struct PolynomialMutationArgs {
    /// A user-defined parameter to control the mutation. This is eta_m in the paper, and it is
    /// suggested its value to be in the [20, 100] range.
    pub index_parameter: f64,
    /// The probability of mutating a parent variable.
    pub variable_probability: f64,
}

impl PolynomialMutationArgs {
    /// Initialise the Polynomial mutation (PM) operator with the default parameters. With a
    /// distribution index or index parameter of 20 and variable probability equal 1 divided by
    /// the number of real variables in the problem (i.e. each variable will have the same
    /// probability of being mutated).
    ///
    /// # Arguments
    ///
    /// * `problem`: THe problem being solved.
    ///
    /// returns: `Self`
    pub fn default(problem: &Problem) -> Self {
        let num_real_vars = problem
            .variables()
            .iter()
            .filter(|(_, v)| v.is_real())
            .count() as f64;
        let variable_probability = 1.0 / num_real_vars;
        Self {
            index_parameter: 20.0,
            variable_probability,
        }
    }
}

/// The Polynomial mutation (PM) operator.
///
/// Adapted from [Deb & Deb (2014)](https://dl.acm.org/doi/10.1504/IJAISC.2014.059280), full
/// text available at <https://www.egr.msu.edu/~kdeb/papers/k2012016.pdf>
pub struct PolynomialMutation {
    /// The user-defined parameter to control the mutation.
    index_parameter: f64,
    /// The probability of mutating a parent variable.
    variable_probability: f64,
}

impl PolynomialMutation {
    /// Initialise the Polynomial mutation (PM) operator. This returns an error if the probability
    /// is outside the [0, 1] range.
    ///
    /// # Arguments
    ///
    /// * `index_parameter`:
    /// * `variable_probability`: The probability of mutating a parent variable.
    ///
    /// returns: `Result<PolynomialMutation, OError>`
    pub fn new(args: PolynomialMutationArgs) -> Result<Self, OError> {
        if !(0.0..=1.0).contains(&args.variable_probability) {
            return Err(OError::MutationOperator(
                "PolynomialMutation".to_string(),
                format!(
                    "The variable probability {} must be a number between 0 and 1",
                    args.variable_probability
                ),
            ));
        }
        Ok(Self {
            index_parameter: args.index_parameter,
            variable_probability: args.variable_probability,
        })
    }
}

impl Mutation for PolynomialMutation {
    fn mutate_offsprings(&self, individual: &Individual) -> Result<Individual, OError> {
        let mut mutated_individual = individual.clone_variables();
        let problem = individual.problem();
        let mut rng = rand::thread_rng();

        // return error if variable is not Real
        if !problem.variables().iter().all(|(_, v)| v.is_real()) {
            return Err(OError::CrossoverOperator(
                "PolynomialMutation".to_string(),
                "The PM operator only works with real variables".to_string(),
            ));
        }

        // do not apply crossover if probability is not reached
        for (var_name, var_type) in problem.variables() {
            if rng.gen_range(0.0..=1.0) <= self.variable_probability {
                let y = individual.get_variable_value(&var_name)?;
                if let (VariableValue::Real(y), VariableType::Real(vt)) = (y, var_type) {
                    let (y_lower, y_upper) = vt.bounds();

                    let delta_y = y_upper - y_lower;
                    let prob = rng.gen_range(0.0..=1.0);

                    // this is delta_l or delta_r
                    let delta = if prob <= 0.5 {
                        let bl = (y - y_lower) / delta_y;
                        let b = 2.0 * prob
                            + (1.0 - 2.0 * prob) * f64::powf(1.0 - bl, self.index_parameter + 1.0);
                        f64::powf(b, 1.0 / (self.index_parameter + 1.0)) - 1.0
                    } else {
                        let bu = (y_upper - y) / delta_y;
                        let b = 2.0 * (1.0 - prob)
                            + 2.0 * (prob - 0.5) * f64::powf(1.0 - bu, self.index_parameter + 1.0);
                        1.0 - f64::powf(b, 1.0 / (self.index_parameter + 1.0))
                    };

                    // adjust the variable
                    let mut new_y = y + delta * delta_y;
                    new_y = f64::min(f64::max(new_y, y_lower), y_upper);
                    mutated_individual.update_variable(&var_name, VariableValue::Real(new_y))?;
                }
            }
        }

        Ok(mutated_individual)
    }
}
