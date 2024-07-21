use log::debug;

use crate::algorithms::nsga3::{
    MIN_DISTANCE, NORMALISED_OBJECTIVE_KEY, REF_POINT, REF_POINT_INDEX,
};
use crate::algorithms::NSGA3;
use crate::core::{DataValue, Individual, OError};
use crate::utils::{argmin, perpendicular_distance};

/// This implements "Algorithm 3" in the paper which associates each individual's normalised
/// objectives to a reference point.
pub struct AssociateToRefPoint<'a> {
    /// The individuals containing the normalised objectives.
    individuals: &'a mut [Individual],
    /// The reference points
    reference_points: &'a [Vec<f64>],
}

impl<'a> AssociateToRefPoint<'a> {
    /// Build the [`AssociateToRefPoint`] structure. This returns an error if the reference point
    /// coordinates are not between 0 and 1.
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals containing the normalised objectives.
    /// * `reference_points`: The reference points to associate the objectives to.
    ///
    /// returns: `Result<Self, OError>`
    pub fn new(
        individuals: &'a mut [Individual],
        reference_points: &'a [Vec<f64>],
    ) -> Result<Self, OError> {
        // check reference point values
        for point in reference_points {
            Self::check_bounds(point)?;
        }

        Ok(Self {
            individuals,
            reference_points,
        })
    }

    /// Associate the individuals to a reference point. If an association is found, this function
    /// stores the distance, the reference point coordinates and reference point index of
    /// [`self.reference_points`] in the individual's data.
    ///
    /// return `Result<(), OError>`
    pub fn calculate(&mut self) -> Result<(), OError> {
        // steps 1-3 are skipped because `reference_points` are already normalised

        // step 4-7
        for ind in self.individuals.iter_mut() {
            // fetch the data
            let data = NSGA3::get_normalised_objectives(ind)?;
            let obj_values = data.as_f64_vec()?;
            // calculate the distances for all reference points
            let d_per = self
                .reference_points
                .iter()
                .map(|ref_point| {
                    perpendicular_distance(ref_point, obj_values).map_err(|e| {
                        OError::AlgorithmRun(
                            "NSGA3-AssociateToRefPoint".to_string(),
                            format!("Cannot calculate vector distance because: {}", e),
                        )
                    })
                })
                .collect::<Result<Vec<f64>, OError>>()?;

            // step 8 - get the reference point with the lowest minimum distance
            let (ri, min_d) = argmin(&d_per);
            ind.set_data(MIN_DISTANCE, DataValue::Real(min_d));
            ind.set_data(
                REF_POINT,
                DataValue::Vector(self.reference_points[ri].clone()),
            );
            ind.set_data(REF_POINT_INDEX, DataValue::USize(ri));
            debug!(
                "Associated objective point {:?} to reference point #{} {:?} - distance = {}",
                ind.get_data(NORMALISED_OBJECTIVE_KEY)?,
                ri,
                self.reference_points[ri],
                min_d
            );
        }

        Ok(())
    }

    /// Check that the values in a reference point are between 0 and 1 (i.e. all the values have
    /// been normalised).
    ///
    /// # Arguments
    ///
    /// * `points`: The reference point coordinates to check.
    ///
    /// returns: `Result<(), OError>`
    fn check_bounds(points: &[f64]) -> Result<(), OError> {
        if points.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(OError::AlgorithmRun(
                "NSGA3-AssociateToRefPoint".to_string(),
                format!(
                    "The values of the reference point {:?} must be between 0 and 1",
                    points,
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;

    use crate::algorithms::AssociateToRefPoint;
    use crate::algorithms::nsga3::{MIN_DISTANCE, NORMALISED_OBJECTIVE_KEY, REF_POINT};
    use crate::core::{DataValue, ObjectiveDirection};
    use crate::core::test_utils::{assert_approx_array_eq, individuals_from_obj_values_dummy};
    use crate::utils::{DasDarren1998, NumberOfPartitions};

    #[test]
    /// Test `AssociateToRefPoint` that calculates the correct distances and reference point
    /// association.
    fn test_simple_association() {
        let das_darren = DasDarren1998::new(3, &NumberOfPartitions::OneLayer(4)).unwrap();
        let ref_points = das_darren.get_weights();

        let dummy_objectives = vec![vec![0.0, 0.0], vec![50.0, 50.0]];
        let mut individuals = individuals_from_obj_values_dummy(
            &dummy_objectives,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
            None,
        );
        // set normalised objectives
        individuals[0].set_data(
            NORMALISED_OBJECTIVE_KEY,
            DataValue::Vector(vec![0.95, 0.15, 0.15]),
        );
        individuals[1].set_data(
            NORMALISED_OBJECTIVE_KEY,
            DataValue::Vector(vec![0.1, 0.9, 0.1]),
        );

        let mut ass = AssociateToRefPoint::new(&mut individuals, &ref_points).unwrap();
        ass.calculate().unwrap();

        // 1st individual
        assert_approx_array_eq(
            individuals[0]
                .get_data(REF_POINT)
                .unwrap()
                .as_f64_vec()
                .unwrap(),
            &[1.0, 0.0, 0.0],
        );
        assert_approx_eq!(
            f64,
            individuals[0]
                .get_data(MIN_DISTANCE)
                .unwrap()
                .as_real()
                .unwrap(),
            0.212132034355,
            epsilon = 0.0001
        );

        // 2nd individual
        assert_approx_array_eq(
            individuals[1]
                .get_data(REF_POINT)
                .unwrap()
                .as_f64_vec()
                .unwrap(),
            &[0.0, 1.0, 0.0],
        );
        assert_approx_eq!(
            f64,
            individuals[1]
                .get_data(MIN_DISTANCE)
                .unwrap()
                .as_real()
                .unwrap(),
            0.1414213562,
            epsilon = 0.0001
        );
    }
}
