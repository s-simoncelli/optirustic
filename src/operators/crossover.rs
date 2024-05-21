use rand::{Rng, RngCore};
use rand::prelude::SliceRandom;

use crate::core::{Individual, OError, VariableType, VariableValue};

/// Struct containing the offsprings from the crossover operation.
#[derive(Debug)]
pub struct CrossoverChildren {
    /// The first generated child.
    pub child1: Individual,
    /// The second generated child.
    pub child2: Individual,
}

/// Trait to define a crossover operator to generate a new child by recombining the genetic
/// material of two parents.
pub trait Crossover {
    /// Generate two children from their parents.
    ///
    /// # Arguments
    ///
    /// * `parent1`: The first parent to use for mating.
    /// * `parent2`: The second parent to use for mating.
    /// * `rng`: The random number generator.
    ///
    /// returns: `Result<CrossoverChildren, OError>`.
    fn generate_offsprings(
        &self,
        parent1: &Individual,
        parent2: &Individual,
        rng: &mut dyn RngCore,
    ) -> Result<CrossoverChildren, OError>;
}

/// Input arguments for [`SimulatedBinaryCrossover`].
#[derive(Clone)]
pub struct SimulatedBinaryCrossoverArgs {
    /// The distribution index for crossover (this is the eta_c in the paper). This directly
    /// control the spread of children. If a large value is selected, the resulting children will
    /// have a higher probability of being close to their parents; a small value generates distant
    /// offsprings.
    pub distribution_index: f64,
    /// The probability that the parents participate in the crossover. If 1.0, the parents always
    /// participate in the crossover. If the probability is lower, then the children are the exact
    /// clones of their parents (i.e. all the variable values do not change).
    pub crossover_probability: f64,
    /// The probability that a variable belonging to both parents is used in the crossover. The
    /// paper uses 0.5, meaning that  each variable in a solution has a 50% chance of changing its
    /// value.
    pub variable_probability: f64,
}

impl Default for SimulatedBinaryCrossoverArgs {
    /// Default parameters for the Simulated Binary Crossover (SBX) with a distribution index of
    /// 15, crossover probability of 1 and variable probability of 0.5.
    fn default() -> Self {
        Self {
            distribution_index: 15.0,
            crossover_probability: 1.0,
            variable_probability: 0.5,
        }
    }
}

/// Simulated Binary Crossover (SBX) operator for bounded real variables.
///
/// Implemented based on:
/// > Kalyanmoy Deb, Karthik Sindhya, and Tatsuya Okabe. 2007. Self-adaptive
/// > simulated binary crossover for real-parameter optimization. In Proceedings of the 9th annual
/// > conference on Genetic and evolutionary computation (GECCO '07). Association for Computing
/// > Machinery, New York, NY, USA, 1187–1194. <https://doi.org/10.1145/1276958.1277190>
///
/// and
/// > the C implementation available at <https://gist.github.com/Tiagoperes/1779d5f1c89bae0cfdb87b1960bba36d>
/// to account for bounded variables.
///
/// See: <https://doi.org/10.1145/1276958.1277190>,
/// full text available at <https://content.wolfram.com/sites/13/2018/02/09-2-2.pdf>. An alternative
/// shorter description of the algorithm is available in [Deb et al. (2007)](https://www.researchgate.net/publication/220742263_Self-adaptive_simulated_binary_crossover_for_real-parameter_optimization).
///
/// # Example
///
/// ```
/// use std::error::Error;
/// use optirustic::core::{BoundedNumber, Individual, Problem, VariableType, VariableValue,
/// Objective, Constraint, ObjectiveDirection, RelationalOperator, EvaluationResult, Evaluator};
/// use optirustic::operators::{Crossover, SimulatedBinaryCrossover, SimulatedBinaryCrossoverArgs};
/// use std::sync::Arc;
/// use rand_chacha::ChaCha8Rng;
/// use rand::SeedableRng;
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     // create a new one-variable problem
///     let objectives = vec![Objective::new("obj1", ObjectiveDirection::Minimise)];
///
///
///     let variables = vec![VariableType::Real(BoundedNumber::new("var1", 0.0, 1000.0)?)];
///     let constraints = vec![Constraint::new("c1", RelationalOperator::EqualTo, 1.0)];
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
///     let problem = Arc::new(Problem::new(objectives, variables, Some(constraints), Box::new(UserEvaluator))?);
///
///     // add new individuals
///     let mut a = Individual::new(problem.clone());
///     a.update_variable("var1", VariableValue::Real(0.2))?;
///     let mut b = Individual::new(problem.clone());
///     b.update_variable("var1", VariableValue::Real(0.8))?;
///
///     // crossover
///     let parameters = SimulatedBinaryCrossoverArgs {
///         distribution_index: 1.0,
///         crossover_probability:1.0,
///         variable_probability:0.5
///     };
///     let sbx = SimulatedBinaryCrossover::new(parameters)?;
///     let mut rng = ChaCha8Rng::from_seed(Default::default());
///     let out = sbx.generate_offsprings(&a, &b, &mut rng)?;
///     println!("{} - {}", out.child1, out.child2);
///     Ok(())
/// }
/// ```
pub struct SimulatedBinaryCrossover {
    /// The distribution index for crossover. This is the eta_c in the paper.
    distribution_index: f64,
    /// The probability that the parents participate in the crossover. If 1.0, the parents always
    /// participate in the crossover.
    crossover_probability: f64,
    /// The probability that a variable belonging to both parents is used in the crossover.
    variable_probability: f64,
}

impl SimulatedBinaryCrossover {
    /// Initialise the Simulated Binary Crossover (SBX) operator for bounded real variables.
    ///
    /// # Arguments
    ///
    /// * `args.`: The operator input parameters. See [`SimulatedBinaryCrossoverArgs`] for a detail
    /// explanation of the parameters.
    ///
    /// returns: `Result<SBX, OError>`
    pub fn new(args: SimulatedBinaryCrossoverArgs) -> Result<Self, OError> {
        if args.distribution_index < 0.0 {
            return Err(OError::CrossoverOperator(
                "SBX".to_string(),
                format!(
                    "The distribution index {} must be a positive number",
                    args.distribution_index
                ),
            ));
        }
        if !(0.0..=1.0).contains(&args.crossover_probability) {
            return Err(OError::CrossoverOperator(
                "SBX".to_string(),
                format!(
                    "The crossover probability {} must be a number between 0 and 1",
                    args.crossover_probability
                ),
            ));
        }
        if !(0.0..=1.0).contains(&args.variable_probability) {
            return Err(OError::CrossoverOperator(
                "SBX".to_string(),
                format!(
                    "The variable probability {} must be a number between 0 and 1",
                    args.variable_probability
                ),
            ));
        }

        Ok(Self {
            distribution_index: args.distribution_index,
            variable_probability: args.variable_probability,
            crossover_probability: args.crossover_probability,
        })
    }

    /// Calculate the betaq coefficient.
    ///
    /// # Arguments
    ///
    /// * `prob`: The probability.
    /// * `alpha`: The alpha coefficient.
    ///
    /// returns: `f64`
    fn betaq(&self, prob: f64, alpha: f64) -> f64 {
        if prob <= (1.0 / alpha) {
            f64::powf(prob * alpha, 1.0 / (self.distribution_index + 1.0))
        } else {
            f64::powf(
                1.0 / (2.0 - prob * alpha),
                1.0 / (self.distribution_index + 1.0),
            )
        }
    }
}

impl Crossover for SimulatedBinaryCrossover {
    fn generate_offsprings(
        &self,
        parent1: &Individual,
        parent2: &Individual,
        rng: &mut dyn RngCore,
    ) -> Result<CrossoverChildren, OError> {
        let mut child1 = parent1.clone_variables();
        let mut child2 = parent2.clone_variables();
        let problem = parent1.problem();

        // return error if variable is not Real
        if !problem.variables().iter().all(|(_, v)| v.is_real()) {
            return Err(OError::CrossoverOperator(
                "SBX".to_string(),
                "The SBX operator only works with real variables".to_string(),
            ));
        }

        // do not apply crossover if probability is not reached
        if rng.gen_range(0.0..=1.0) <= self.crossover_probability {
            for (var_name, var_type) in problem.variables() {
                // each variable in a solution has a `self.variable_probability` chance of changing
                // its value
                if rng.gen_range(0.0..=1.0) > self.variable_probability {
                    continue;
                }

                // do not process non-real variables
                let v1 = parent1.get_variable_value(&var_name)?;
                let v2 = parent2.get_variable_value(&var_name)?;
                if let (VariableValue::Real(v1), VariableValue::Real(v2), VariableType::Real(vt)) =
                    (v1, v2, var_type)
                {
                    // do not perform crossover if variables have the same value
                    if f64::abs(v1 - v2) < f64::EPSILON {
                        continue;
                    }

                    // get the lowest value between the two parent
                    let (y1, y2) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                    let (y_lower, y_upper) = vt.bounds();
                    let delta_y = y2 - y1;
                    let prob = rng.gen_range(0.0..=1.0);

                    // first child
                    let beta = 1.0 + (2.0 * (y1 - y_lower) / delta_y);
                    let alpha = 2.0 - f64::powf(beta, -(self.distribution_index + 1.0));
                    let mut new_v1 = 0.5 * ((y1 + y2) - self.betaq(prob, alpha) * delta_y);
                    // make sure value is within bounds
                    new_v1 = f64::min(f64::max(new_v1, y_lower), y_upper);

                    // second child
                    let beta = 1.0 + (2.0 * (y_upper - y2) / delta_y);
                    let alpha = 2.0 - f64::powf(beta, -(self.distribution_index + 1.0));
                    let mut new_v2 = 0.5 * ((y1 + y2) + self.betaq(prob, alpha) * delta_y);
                    // make sure value is within bounds
                    new_v2 = f64::min(f64::max(new_v2, y_lower), y_upper);

                    // randomly swap the values
                    if matches!([0, 1].choose(rng).unwrap(), 0) {
                        (new_v1, new_v2) = (new_v2, new_v1);
                    }

                    // update the children
                    child1.update_variable(&var_name, VariableValue::Real(new_v1))?;
                    child2.update_variable(&var_name, VariableValue::Real(new_v2))?;
                }
            }
        }

        Ok(CrossoverChildren { child1, child2 })
    }
}

#[cfg(test)]
mod test {
    use crate::operators::{SimulatedBinaryCrossover, SimulatedBinaryCrossoverArgs};

    #[test]
    /// Check that the input arguments to SBX operator are valid.
    fn test_new_panic() {
        assert!(SimulatedBinaryCrossover::new(SimulatedBinaryCrossoverArgs {
            distribution_index: -2.0,
            crossover_probability: 1.0,
            variable_probability: 0.5,
        })
        .is_err());
        assert!(SimulatedBinaryCrossover::new(SimulatedBinaryCrossoverArgs {
            distribution_index: 1.0,
            crossover_probability: 2.0,
            variable_probability: 0.5,
        })
        .is_err());
        assert!(SimulatedBinaryCrossover::new(SimulatedBinaryCrossoverArgs {
            distribution_index: 1.0,
            crossover_probability: 1.0,
            variable_probability: -0.5,
        })
        .is_err());
    }
}
