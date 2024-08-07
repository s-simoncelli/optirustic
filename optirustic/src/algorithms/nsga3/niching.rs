use std::collections::HashMap;

use log::debug;
use rand::prelude::SliceRandom;
use rand::RngCore;

use crate::algorithms::nsga3::{
    MIN_DISTANCE, NORMALISED_OBJECTIVE_KEY, REF_POINT, REF_POINT_INDEX,
};
use crate::core::{DataValue, Individual, OError, Population};
use crate::utils::{argmin_by, index_of};

/// This implements "Algorithm 4" in the paper which adds individuals from the last front to the new
/// population based on the reference point association and minimum distance.
pub(crate) struct Niching<'a> {
    /// The population being created at the current evolution with the new selected individuals.
    /// This is `$P_{t+1}$` from the paper and is populated with individuals from [`self.potential_individuals`].
    selected_individuals: &'a mut Population,
    /// Individuals from the last front $F_l$ to add to [`self.selected_individuals`] based on reference
    /// point association and minimum distance.
    potential_individuals: &'a mut Vec<Individual>,
    /// The number of individuals to add to [`self.selected_individuals`] to complete the evolution.
    missing_item_count: usize,
    /// The map mapping the reference point index to the number of individuals already associated in
    /// [`self.selected_individuals`]
    rho_j: &'a mut HashMap<usize, usize>,
    /// The random number generator
    rng: &'a mut Box<dyn RngCore>,
}

impl<'a> Niching<'a> {
    /// Niching algorithm.
    ///
    /// # Arguments
    ///
    /// * `selected_individuals`: The population P_{t+1} without the last front. This will be
    ///    populated with individuals from `potential_individuals`.
    /// * `potential_individuals`: The potential individuals from the last front.
    /// * `number_of_individuals_to_add`: The number of individuals to add to `selected_individuals`
    ///    from `potential_individuals`.
    /// * `rho_j`: The map containing the reference point indexes as keys and the number of associated
    ///    points from P_{t+1}.
    /// * `rng`: The random number generator.
    ///
    /// returns: `Result<Niching, OError>`
    pub fn new(
        selected_individuals: &'a mut Population,
        potential_individuals: &'a mut Vec<Individual>,
        number_of_individuals_to_add: usize,
        rho_j: &'a mut HashMap<usize, usize>,
        rng: &'a mut Box<dyn RngCore>,
    ) -> Result<Self, OError> {
        let name = "NSGA3-Niching".to_string();
        if rho_j.is_empty() {
            return Err(OError::AlgorithmRun(
                name,
                "The rho_j set is empty".to_string(),
            ));
        }
        if potential_individuals.len() < number_of_individuals_to_add {
            return Err(OError::AlgorithmRun(
                name,
                format!("The number of individuals to add ({number_of_individuals_to_add}) is larger than the number of potential individuals ({})", potential_individuals.len()),
            ));
        }

        Ok(Self {
            selected_individuals,
            potential_individuals,
            missing_item_count: number_of_individuals_to_add,
            rho_j,
            rng,
        })
    }

    /// Add new individuals to the population. This updates [`self.new_population`] by draining
    /// items from [`self.potential_individuals`]. Reference points not associated with any point in
    /// [`self.potential_individuals`] and excluded from the current evolution are removed from
    /// [`self.rho_j`].
    ///
    /// return: `Result<(), OError>`
    pub fn calculate(&mut self) -> Result<(), OError> {
        let mut k = 1;
        let name = "NSGA3-Niching".to_string();
        debug!(
            "Number of individuals to choose {}",
            self.missing_item_count
        );
        while k <= self.missing_item_count {
            debug!(
                "Adding point {k}/{} to new population",
                self.missing_item_count
            );

            // step 3 - select the reference point with the minimum rho_j counter. Reference points
            // that have no association with individuals in F_l (Z_r = Z_r/{j_hat}, step 15) are
            // excluded by removing them from rho_j later
            let min_rho_j = *self
                .rho_j
                .iter()
                .min_by(|(_, v1), (_, v2)| v1.cmp(v2))
                .unwrap()
                .1;
            debug!("min_rho_j = {min_rho_j}");

            // step 3 - collect all reference point indexes j with minimum rho_j
            let j_min_set: Vec<usize> = self
                .rho_j
                .iter()
                .filter_map(|(ref_index, ref_counter)| {
                    if *ref_counter == min_rho_j {
                        Some(*ref_index)
                    } else {
                        None
                    }
                })
                .collect();

            // step 4 - get reference point with minimum association counter
            let j_hat = match j_min_set.len() {
                0 => return Err(OError::AlgorithmRun(name, "Empty j_min_set".to_string())),
                1 => *j_min_set.first().unwrap(),
                // select point randomly when set size is > 1
                _ => *j_min_set.choose(self.rng.as_mut()).unwrap(),
            };
            debug!("Selected reference point j_hat=#{j_hat}");

            // step 5 - individuals in F_j linked to current reference point index j_hat
            let i_j: Vec<&Individual> = self
                .potential_individuals
                .iter()
                .filter(|ind| ind.get_data(REF_POINT_INDEX).unwrap() == DataValue::USize(j_hat))
                .collect();
            debug!(
                "Found {} potential individuals associated with it",
                i_j.len()
            );

            // step 6 - select points from front F_l
            if !i_j.is_empty() {
                // let (_, ind) = argmin_by(&i_j, |(_, ind)| {
                //     ind.get_data(MIN_DISTANCE).unwrap().as_real().unwrap()
                // })
                // .unwrap();
                // let new_ind_index = index_of(self.potential_individuals, ind);
                // let method = "min_distance";

                let (new_ind_index, method) = if min_rho_j == 0 {
                    // step 7 - no point from P_{t+1} is associated with the selected reference point
                    // j_hat. There's at least one point from F_l that can be linked (I_j is not empty)

                    // step 8 - find individual in F_l with the shortest distance
                    let (_, ind) = argmin_by(&i_j, |(_, ind)| {
                        ind.get_data(MIN_DISTANCE).unwrap().as_real().unwrap()
                    })
                    .unwrap();
                    let new_ind_index = index_of(self.potential_individuals, ind);
                    (new_ind_index, "min_distance")
                } else {
                    // step 10 - choose random point from F_l
                    let ind = i_j.choose(self.rng).unwrap();
                    let new_ind_index = index_of(self.potential_individuals, ind);
                    (new_ind_index, "random")
                };

                // step 12a - mark reference point as associated to a new F_l's individual
                *self.rho_j.get_mut(&j_hat).unwrap() += 1;

                // step 12b - Add new individual and remove it from F_l
                match new_ind_index {
                    None => {
                        return Err(OError::AlgorithmRun(
                            name,
                            "Cannot find individual's index".to_string(),
                        ))
                    }
                    Some(index) => {
                        let ind = self.potential_individuals.remove(index);
                        debug!(
                            "Added individual #{index} {:?} to population ({method}) - reference point #{j_hat} {:?}",
                            ind.get_data(NORMALISED_OBJECTIVE_KEY)?,
                            ind.get_data(REF_POINT),
                        );
                        self.selected_individuals.add_individual(ind);

                        // step 13
                        k += 1;
                    }
                }
            } else {
                // step 15 - no point in F_l is associated with reference point indexed by j_hat.
                // j_hat will have no linked individual at this evolution. Exclude it.
                debug!("Excluding ref point index {j_hat} - no candidates associated with it");
                self.rho_j.remove(&j_hat);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::algorithms::nsga3::niching::Niching;
    use crate::algorithms::nsga3::{MIN_DISTANCE, REF_POINT_INDEX};
    use crate::core::test_utils::individuals_from_obj_values_dummy;
    use crate::core::utils::get_rng;
    use crate::core::{DataValue, Individual, ObjectiveDirection, Population};

    #[test]
    /// Check niching that (1) adds point with min distance when there reference point is not
    /// already associated with an objective; (2) reference points not linked to potential individuals
    /// are excluded from the algorithm.
    fn test_niching_rho0() {
        // create dummy population with 4 individuals
        let dummy_objectives = vec![vec![0.0, 0.0]; 2];
        let mut individuals = individuals_from_obj_values_dummy(
            &dummy_objectives,
            &[ObjectiveDirection::Minimise; 2],
            None,
        );
        let problem = individuals[0].problem().clone();
        let mut rho_j: HashMap<usize, usize> = HashMap::new();

        // link 2 individuals to 2 out of 4 reference points
        individuals[0].set_data(REF_POINT_INDEX, DataValue::USize(0));
        individuals[0].set_data(MIN_DISTANCE, DataValue::Real(0.1));
        rho_j.entry(0).or_insert(1);

        individuals[1].set_data(REF_POINT_INDEX, DataValue::USize(1));
        individuals[1].set_data(MIN_DISTANCE, DataValue::Real(0.2));
        rho_j.entry(1).or_insert(1);
        let mut pop = Population::new_with(individuals);

        // potential individuals - both are linked to ref_point #3 but ind_3 is closer
        let mut ind_3 = Individual::new(problem.clone());
        ind_3.set_data(REF_POINT_INDEX, DataValue::USize(2));
        ind_3.set_data(MIN_DISTANCE, DataValue::Real(0.4));

        let mut ind_4 = Individual::new(problem);
        ind_4.set_data(REF_POINT_INDEX, DataValue::USize(2));
        ind_4.set_data(MIN_DISTANCE, DataValue::Real(0.9));

        // counter is 0 for all other ref_points
        rho_j.entry(2).or_insert(0);
        rho_j.entry(3).or_insert(0);
        let mut potential_individuals = vec![ind_3, ind_4];
        let selected_ind = potential_individuals[0].clone();

        let mut rng = get_rng(Some(1));
        let mut n = Niching::new(
            &mut pop,
            &mut potential_individuals,
            1,
            &mut rho_j,
            &mut rng,
        )
        .unwrap();
        n.calculate().unwrap();
        // let excluded = n.excluded_ref_point_index;

        // counter for ref_point #3 has increased
        assert_eq!(rho_j[&2_usize], 1_usize);
        // 3rd individual is added to the population
        assert_eq!(pop.len(), 3);
        assert_eq!(pop.individual(2).unwrap(), &selected_ind);
        // 4th reference point should be excluded because has no association
        // NOTE: do not check this due to random ref_point selection (if point 4 is picked first,
        // this is excluded otherwise it will not).
        // assert!(
        //     !rho_j.contains(&3_usize),
        //     "excluded = {:?} does not contain ref_point #4",
        //     rho_j
        // );
    }

    #[test]
    /// Check niching that adds point with min distance when there reference point is already
    /// associated with another objective.
    fn test_niching_rho1() {
        // create dummy population with 4 individuals
        let dummy_objectives = vec![vec![0.0, 0.0]; 2];
        let mut individuals = individuals_from_obj_values_dummy(
            &dummy_objectives,
            &[ObjectiveDirection::Minimise; 2],
            None,
        );
        let problem = individuals[0].problem().clone();
        let mut rho_j: HashMap<usize, usize> = HashMap::new();

        // link 2 individuals to 2 out of 4 reference points
        individuals[0].set_data(REF_POINT_INDEX, DataValue::USize(0));
        individuals[0].set_data(MIN_DISTANCE, DataValue::Real(0.1));
        rho_j.entry(0).or_insert(1);

        individuals[1].set_data(REF_POINT_INDEX, DataValue::USize(1));
        individuals[1].set_data(MIN_DISTANCE, DataValue::Real(0.2));
        rho_j.entry(1).or_insert(1);
        let mut pop = Population::new_with(individuals);

        // potential individuals - both are linked to ref_point #2 but ind_4 is closer
        let mut ind_3 = Individual::new(problem.clone());
        ind_3.set_data(REF_POINT_INDEX, DataValue::USize(1));
        ind_3.set_data(MIN_DISTANCE, DataValue::Real(99.0));

        let mut ind_4 = Individual::new(problem);
        ind_4.set_data(REF_POINT_INDEX, DataValue::USize(1));
        ind_4.set_data(MIN_DISTANCE, DataValue::Real(0.9));

        // counter is 0 for all other ref_points
        rho_j.entry(2).or_insert(0);
        let mut potential_individuals = vec![ind_3, ind_4];
        let selected_ind = potential_individuals[1].clone();

        let mut rng = get_rng(Some(1));
        let mut n = Niching::new(
            &mut pop,
            &mut potential_individuals,
            1,
            &mut rho_j,
            &mut rng,
        )
        .unwrap();
        n.calculate().unwrap();

        // counter for ref_point #3 has increased
        assert_eq!(rho_j[&1_usize], 2_usize);
        // 3rd individual is added to the population
        assert_eq!(pop.len(), 3);
        assert_eq!(pop.individual(2).unwrap(), &selected_ind);
    }
}
