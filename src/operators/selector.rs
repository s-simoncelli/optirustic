use std::marker::PhantomData;

use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::core::{Individual, Population};
use crate::operators::{BinaryComparisonOperator, PreferredSolution};

/// A trait implementing methods to choose individuals from a population for reproduction.
pub trait Selector<'a> {
    /// Select a number of individuals from the population equal to `number_of_winners`.
    ///
    /// # Arguments
    ///
    /// * `population`: The population.
    /// * `number_of_winners`: The number of winners to select.
    ///
    /// returns: `Result<[&Individual], String>`
    fn select(
        &'a self,
        population: &'a Population,
        number_of_winners: usize,
    ) -> Result<Vec<&'a Individual>, String> {
        let mut winners: Vec<&Individual> = Vec::new();
        for _ in 0..number_of_winners {
            winners.push(self.select_fit_individual(population)?);
        }
        Ok(winners)
    }

    /// Select the fittest individual from the population.
    ///
    /// # Arguments
    ///
    /// * `population`: The list of individuals.
    ///
    /// returns: `Result<&Individual, String>`
    fn select_fit_individual(
        &'a self,
        population: &'a Population,
    ) -> Result<&'a Individual, String>;
}

/// Tournament selection method between multiple competitors for choosing individuals from a
/// population for reproduction. `number_of_competitors` individuals are randomly selected from the
/// population, then the most fit becomes a parent based on the provided `fitness` function.
/// More tournaments may be run to select more individuals.
pub struct TournamentSelector<Operator: BinaryComparisonOperator> {
    /// The number of competitors in each tournament. For example, 2 to run a binary tournament.
    number_of_competitors: usize,
    /// The function to use to assess the fitness and determine which individual wins a tournament.
    _fitness_function: PhantomData<Operator>,
}

impl<Operator: BinaryComparisonOperator> TournamentSelector<Operator> {
    /// Create a new tournament.
    ///
    /// # Arguments
    ///
    /// * `number_of_competitors`: The number of competitors in the tournament. Default to 2
    /// individuals.
    ///
    /// returns: `TournamentSelector`
    pub fn new(number_of_competitors: usize) -> Self {
        Self {
            _fitness_function: PhantomData::<Operator>,
            number_of_competitors,
        }
    }
}

impl<'a, Operator: BinaryComparisonOperator> Selector<'a> for TournamentSelector<Operator> {
    /// Select the fittest individual from the population.
    ///
    /// # Arguments
    ///
    /// * `population`:The population with the solutions.
    ///
    /// returns: `Result<&Individual, String>`
    fn select_fit_individual(
        &'a self,
        population: &'a Population,
    ) -> Result<&'a Individual, String> {
        if population.is_empty() {
            return Err("The population is empty and no individual can be selected".to_string());
        }
        if population.size() < self.number_of_competitors {
            return Err(format!("The population size ({}) is smaller than the number of competitors needed in the tournament ({})", population.size(), self.number_of_competitors));
        }

        let individuals = population.individuals();
        let problem = individuals.first().unwrap().problem();
        let mut rng = thread_rng();
        let mut winner = individuals.choose(&mut rng).unwrap();

        for _ in 0..self.number_of_competitors {
            let potential_winner = individuals.choose(&mut rng).unwrap();
            let preferred_sol = Operator::compare(problem, winner, potential_winner)?;
            if preferred_sol == PreferredSolution::Second {
                winner = potential_winner;
            } else if preferred_sol == PreferredSolution::MutuallyPreferred {
                // randomly select winner
                winner = [winner, potential_winner].choose(&mut rng).unwrap();
            }
        }

        Ok(winner)
    }
}
