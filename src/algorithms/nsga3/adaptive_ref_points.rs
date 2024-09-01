use std::collections::HashMap;

use log::debug;

use crate::core::OError;
use crate::utils::{vector_min, DasDarren1998, NumberOfPartitions};

/// This implements the adaptive reference point procedure Jain and Deb (2013) in Section VII. This
/// will identify non-useful points and re-position them to try to allocate one point for each
/// Pareto-optimal solution. This archived in two steps:
///
/// 1) **Addition of new reference points:** for each crowded point, where the niche counter rho_j
///    is larger or equal to 2, three new points are added around the reference point. The points
///    are centred around the original point with the same gap.
///    The new points are accepted if they lie in the 1st quadrant (i.e. all their coordinates are
///    between 0 and 1) and they do not overlap with any existing reference point.
/// 2) **Deletion of existing reference points:** if all reference point have a niche counter equal
///    to 1 (ideal case), all new included points with rho_j=0 are deleted. Original points with
///    no association and new points with rho_j=1 are preserved.
///
/// Implemented based on:
/// > Jain, Himanshu & Deb, Kalyanmoy. (2014). An Evolutionary Many-Objective Optimization
/// > Algorithm Using Reference-Point Based Non dominated Sorting Approach, Part II: Handling
/// > Constraints and Extending to an Adaptive Approach. Evolutionary Computation, IEEE
/// > Transactions on. 18. 602-622. <doi.org/10.1109/TEVC.2013.2281534>.
pub(crate) struct AdaptiveReferencePoints<'a> {
    /// The vector of reference points. Their position will change and new points will be added or
    /// deleted depending on the Pareto front evolution.
    reference_points: &'a mut Vec<Vec<f64>>,
    /// The map mapping the reference point index to the number of associated individuals.
    rho_j: &'a mut HashMap<usize, usize>,
    /// The number of original reference points. This is used to identify whether a point in
    /// `reference_points` and `rho_j` is an original one, created when the evolution started.
    number_of_or_reference_points: usize,
    /// The three points to add around an original reference point. They lie on the plane through
    /// the origin
    new_points_set: Vec<Vec<f64>>,
}

impl<'a> AdaptiveReferencePoints<'a> {
    /// Initialise the adaptive algorithm.
    ///
    /// # Arguments
    ///
    /// * `reference_points`: The vector of reference points.
    /// * `rho_j`: The map mapping the reference point index to the number of associated
    ///    individuals.
    /// * `number_of_or_reference_points`: The number of original reference points.
    ///
    /// returns: `Result<AdaptiveReferencePoints, OError>`
    pub fn new(
        reference_points: &'a mut Vec<Vec<f64>>,
        rho_j: &'a mut HashMap<usize, usize>,
        number_of_or_reference_points: usize,
    ) -> Result<Self, OError> {
        let number_of_objectives = reference_points.first().unwrap().len();

        // measure the minimum gap between the points. This equal to 1 divided by the number of
        // partitions for one layer. With two layer, this gives the minimum gap of the two layers.
        let gap = (0..reference_points.len() - 1)
            .flat_map(|p_id| {
                (0..3)
                    .filter_map(|c_id| {
                        let d = f64::abs(
                            reference_points[p_id][c_id] - reference_points[p_id + 1][c_id],
                        );
                        if d > 0.0 {
                            Some(d)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let gap = vector_min(&gap)?;

        // create the 3 new points with the gap. These can be shifted with respect to the original
        // reference point later
        let ds = DasDarren1998::new(number_of_objectives, &NumberOfPartitions::OneLayer(1))?;

        // scale them to have the same original gap
        let mut new_points_set = ds.get_weights();
        new_points_set = new_points_set
            .iter()
            .map(|new_p| new_p.iter().map(|new_c| new_c * gap).collect())
            .collect();

        // shift them with respect to the plane centroid so that the plane they lie on pass through
        // the origin
        let centroid: Vec<f64> = (0..number_of_objectives)
            .map(|c_idx| {
                new_points_set.iter().map(|v| v[c_idx]).sum::<f64>() / new_points_set.len() as f64
            })
            .collect();
        new_points_set = new_points_set
            .iter()
            .map(|new_p| {
                new_p
                    .iter()
                    .zip(&centroid)
                    .map(|(coord, ci)| coord - ci)
                    .collect()
            })
            .collect();

        Ok(AdaptiveReferencePoints {
            reference_points,
            rho_j,
            number_of_or_reference_points,
            new_points_set,
        })
    }

    /// Add and remove new reference points.
    pub fn calculate(&mut self) -> Result<(), OError> {
        self.add()?;
        self.delete();

        Ok(())
    }

    /// Add new reference points if needed.
    fn add(&mut self) -> Result<(), OError> {
        // addition is performed on original reference points only.
        let mut all_new_points = vec![];
        for (ref_point_index, counter) in self.rho_j.iter() {
            if self.is_original_ref_point(*ref_point_index) && *counter >= 2 {
                // shift the simplex with the three points to be centred around the original
                // reference point
                let new_points: Vec<Vec<f64>> = self
                    .new_points_set
                    .iter()
                    .map(|new_p| {
                        // coordinates
                        new_p
                            .iter()
                            .zip(&self.reference_points[*ref_point_index])
                            .map(|(new_c, old_c)| new_c + old_c)
                            .collect()
                    })
                    .collect();

                debug!(
                    "Selected new ref points {:?} for #{ref_point_index} = {:?}",
                    new_points, &self.reference_points[*ref_point_index]
                );
                all_new_points.push(new_points);
            }
        }

        // add the points to the ref point list and niche counter
        for new_points in all_new_points {
            for point in new_points {
                // skip if the point already exists or is outside the 1st quadrant
                if self.has_point(&point, None) {
                    debug!("Disregarded already existing ref point {:?}", point);
                    continue;
                }
                if point.iter().any(|c| *c < 0.0 || *c > 1.0) {
                    debug!("Disregarded outside-range ref point {:?}", point);
                    continue;
                }

                debug!("Added new ref points {:?}", point);
                self.reference_points.push(point);
                let new_index = self.reference_points.len();
                self.rho_j.insert(new_index, 0);
            }
        }

        Ok(())
    }

    /// Delete the additional reference points.
    fn delete(&mut self) {
        // Count number of points with a perfect association to an individual (niche counter is 1)
        let perfect_assoc_counter: usize = self
            .rho_j
            .iter()
            .map(|(_, counter)| if *counter == 1 { 1 } else { 0 })
            .sum();

        println!("perfect_assoc_counter={perfect_assoc_counter}");
        // delete new points without association
        if perfect_assoc_counter == self.number_of_or_reference_points {
            let mut points_to_delete = vec![];
            for (ref_point_idx, point) in self.reference_points.iter().enumerate() {
                // ignore original reference points
                if self.is_original_ref_point(ref_point_idx) {
                    continue;
                }
                // delete new point from map
                if self.rho_j[&ref_point_idx] == 0 {
                    points_to_delete.push(point.clone());
                    self.rho_j.remove(&ref_point_idx);
                }
            }

            println!("points_to_delete={:?}", points_to_delete);
            self.reference_points
                .retain(|v| !points_to_delete.contains(&v));
        }
    }

    /// Check whether the reference point index identifies a original point created when the
    /// evolution started.
    ///
    /// # Arguments
    ///  * `ref_point_index`: The reference point index.
    ///
    /// return: `bool`
    fn is_original_ref_point(&self, ref_point_index: usize) -> bool {
        ref_point_index < self.number_of_or_reference_points
    }

    /// Check if the reference point set has already the new `point`. This returns `true` if all
    /// `point`'s coordinates are close to a reference point within the `tolerance`. This is done
    /// due to rounding errors when the new reference points are scaled and translated.
    ///
    /// # Arguments
    ///
    /// * `point`: The new point to check.
    /// * `tolerance`: The tolerance. Defaults to `10^-6`.
    ///
    /// returns: `bool`
    pub fn has_point(&self, point: &[f64], tolerance: Option<f64>) -> bool {
        let tolerance = tolerance.unwrap_or(0.000001);
        for ref_point in self.reference_points.iter() {
            if ref_point
                .iter()
                .zip(point)
                .all(|(r, p)| (r - p).abs() < tolerance)
            {
                return true;
            }
        }
        return false;
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::panic;

    use crate::algorithms::nsga3::adaptive_ref_points::AdaptiveReferencePoints;
    use crate::core::test_utils::assert_approx_array_eq;
    use crate::utils::{DasDarren1998, NumberOfPartitions};

    #[test]
    /// Add new points around the reference points
    fn test_ref_point_addition() {
        let ds = DasDarren1998::new(3, &NumberOfPartitions::OneLayer(5)).unwrap();

        let test_data = HashMap::from([
            // 2 points have negative coordinates
            (
                3_usize,
                (1_usize, vec![vec![0.13333333, 0.53333333, 0.33333333]]),
            ),
            // all 3 points are added
            (
                12,
                (
                    3,
                    vec![
                        vec![0.33333333, 0.13333333, 0.53333333],
                        vec![0.33333333, 0.33333333, 0.33333333],
                        vec![0.53333333, 0.13333333, 0.33333333],
                    ],
                ),
            ),
        ]);

        for (sc_id, (ref_point_index, results)) in test_data.iter().enumerate() {
            let mut ref_points = ds.get_weights();
            let mut rho_j = HashMap::new();
            for point_idx in 0..ref_points.len() {
                rho_j.insert(point_idx, 0);
            }
            *rho_j.get_mut(&ref_point_index).unwrap() = 2_usize;
            let counter = ref_points.len();

            let mut a = AdaptiveReferencePoints::new(&mut ref_points, &mut rho_j, counter).unwrap();
            a.calculate().unwrap();

            // check point counter
            assert_eq!(ref_points.len(), counter + results.0);
            assert_eq!(rho_j.len(), ref_points.len());

            // check expected points
            for (pos, expected) in results.1.iter().enumerate() {
                let result = panic::catch_unwind(|| {
                    assert_approx_array_eq(
                        &ref_points[ref_points.len() - (results.1.len()) + pos],
                        &expected,
                        None,
                    );
                });
                match result {
                    Ok(_) => {}
                    Err(e) => match e.downcast::<String>() {
                        Ok(v) => panic!(
                            "Scenario {} failed because:\n {:?}.\n Reference points were: {:?}",
                            sc_id + 1,
                            v,
                            ref_points
                        ),
                        _ => {}
                    },
                }
            }
        }
    }

    #[test]
    /// Create more points, but some overlaps and are not added twice
    fn test_ref_point_addition_overlap() {
        let ds = DasDarren1998::new(3, &NumberOfPartitions::OneLayer(5)).unwrap();
        let mut ref_points = ds.get_weights();
        let mut rho_j = HashMap::new();
        for point_idx in 0..ref_points.len() {
            rho_j.insert(point_idx, 0);
        }
        assert_approx_array_eq(&ref_points[7], &[0.2, 0.2, 0.6], Some(0.0));
        *rho_j.get_mut(&7).unwrap() = 2_usize;

        assert_approx_array_eq(&ref_points[12], &[0.4, 0.2, 0.4], Some(0.0));
        *rho_j.get_mut(&12).unwrap() = 2_usize;

        let counter = ref_points.len();

        let mut a = AdaptiveReferencePoints::new(&mut ref_points, &mut rho_j, counter).unwrap();
        a.calculate().unwrap();

        // one point ([0.33333333 0.13333333 0.53333333]) is in common
        assert_eq!(ref_points.len(), counter + 5);
        assert_eq!(rho_j.len(), ref_points.len());
    }

    #[test]
    fn test_ref_point_deletion() {
        let ds = DasDarren1998::new(3, &NumberOfPartitions::OneLayer(5)).unwrap();
        let mut ref_points = ds.get_weights();
        let or_counter = ref_points.len();

        let mut rho_j = HashMap::new();
        // all original points have one association except one
        for point_idx in 0..ref_points.len() {
            rho_j.insert(point_idx, 1);
        }
        rho_j.insert(11, 0);

        // add new points
        ref_points.push(vec![0.33, 0.13, 0.53]); // 21
        ref_points.push(vec![0.33, 0.33, 0.33]); // 22
        rho_j.insert(21, 0);
        rho_j.insert(22, 1);

        println!("rho_j={:?}", rho_j);
        // TODO point 21 will be deleted
        let counter = ref_points.len();
        println!("{:?}", counter);
        let mut a = AdaptiveReferencePoints::new(&mut ref_points, &mut rho_j, or_counter).unwrap();
        a.calculate().unwrap();

        // one additional point is preserved
        assert_eq!(a.reference_points.len(), or_counter + 1);

        // point 21 is deleted
        assert_eq!(ref_points.contains(&vec![0.33, 0.13, 0.53]), false);
        assert_eq!(rho_j.contains_key(&21), false);

        // point 22 still present along with all original ref points
        assert_eq!(ref_points.contains(&vec![0.33, 0.33, 0.33]), true);
        assert_eq!(rho_j.contains_key(&22), true);
        assert_eq!(rho_j.contains_key(&11), true);
    }
}
