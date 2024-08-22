use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::core::OError;

/// Calculate the binomial coefficient. This gives the number of `k`-subsets possible out of a
/// set of `n` distinct items. See <https://mathworld.wolfram.com/BinomialCoefficient.html>. Code
/// adapted from <https://blog.plover.com/math/choose.html>.
///
/// # Arguments
///
/// * `n`: The number of possibilities.
/// * `k`: The number of outcomes.
///
/// returns: `u64`
fn binomial_coefficient(mut n: u64, k: u64) -> u64 {
    let mut r: u64 = 1;
    if k > n {
        0
    } else {
        for d in 1..=k {
            r *= n;
            n -= 1;
            r /= d;
        }
        r
    }
}

/// Define the number of partitions for the two layers.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TwoLayerPartitions {
    /// This is the number of partitions to use in the boundary layer.
    pub boundary_layer: usize,
    /// This is the number of partitions to use in the inner layer.
    pub inner_layer: usize,
    /// Control the size of the inner layer. This defaults to 0.5 which means that the maximum points
    /// on each objectives axis will be located at 0.5 instead of 1 (as in the boundary layer).
    pub scaling: Option<f64>,
}

/// Define the number of partitions to use to generate the reference points. You can create:
///  - 1 layer or set of points with a constant uniform gaps with [`NumberOfPartitions::OneLayer`].
///  - 2 layers of points with each layer having a different gap with [`NumberOfPartitions::TwoLayers`].
///    Use this approach if you are trying to solve a problem with many objectives (4 or more) and
///    want to reduce the number of reference points to use. Using two layers allows (1) setting a
///    smaller number of reference points, (2) controlling the point density in the inner area and
///    (3) ensure a well-spaced point distribution.
#[derive(Serialize, Clone, Deserialize, Debug)]
pub enum NumberOfPartitions {
    /// Create only one layer of points by specifying the number of uniform gaps between two
    /// consecutive points along all objective axis on the hyper-plane.
    OneLayer(usize),
    /// Create two sets of points with two different gap values. The [`DasDarren1998`] approach will
    /// be used to generate the two sets independently which will be then merged into one final set.
    TwoLayers(TwoLayerPartitions),
}

/// Derive the reference points or weights using the methodology suggested in Section 5.2 in the
/// Das & Dennis (1998) paper:
///
/// > Indraneel Das and J. E. Dennis. Normal-Boundary Intersection: A New Method for Generating the
/// > Pareto Surface in Nonlinear Multicriteria Optimization Problems. SIAM Journal on Optimization.
/// > 1998 8:3, 631-657. <https://doi.org/10.1137/S1052623496307510>
///
/// # Examples
/// ## One layer of reference points
/// ```rust
#[doc = include_str!("../../examples/reference_points_1layer.rs")]
/// ```
/// ## Two layers of reference points, each with a different gap
/// ```rust
#[doc = include_str!("../../examples/reference_points_2layers.rs")]
/// ```
pub struct DasDarren1998 {
    /// The number of problem objectives.
    number_of_objectives: usize,
    /// The number of uniform gaps between two consecutive points along all objective axis on the
    /// hyperplane. With this option you can create one or two layer of points with different spacing.
    number_of_partitions: NumberOfPartitions,
}

impl DasDarren1998 {
    /// Initialise the Das & Darren approach to calculate reference points or weights.
    ///
    /// # Arguments
    ///
    /// * `number_of_objectives`: The number of problem objectives.
    /// * `number_of_partitions`: The number of uniform gaps between two consecutive points along
    ///    all objective axis on the hyperplane.
    ///
    /// returns: `Result<DasDarren1998, OError>`
    pub fn new(
        number_of_objectives: usize,
        number_of_partitions: &NumberOfPartitions,
    ) -> Result<Self, OError> {
        match &number_of_partitions {
            NumberOfPartitions::OneLayer(_) => {}
            NumberOfPartitions::TwoLayers(layers) => {
                if let Some(scaling) = layers.scaling {
                    if scaling < f64::EPSILON {
                        return Err(OError::Generic(
                            "The inner layer scaling factor must be larger 0".to_string(),
                        ));
                    }
                }
            }
        }

        Ok(DasDarren1998 {
            number_of_objectives,
            number_of_partitions: number_of_partitions.clone(),
        })
    }

    /// Determine the number of reference points on the `self::number_of_objectives`-dimensional
    /// unit simplex with `self.number_of_partitions` gaps from Section 5.2 of the
    /// [Das & Dennis's paper](https://doi.org/10.1137/S1052623496307510).
    ///
    /// returns: `u64`. The number of reference points.
    pub fn number_of_points(&self) -> u64 {
        match &self.number_of_partitions {
            NumberOfPartitions::OneLayer(number_of_partitions) => {
                // Binomial coefficient of M + p - 1 and p, where M = self.number_of_objectives and
                // p = self.number_of_partitions
                binomial_coefficient(
                    self.number_of_objectives as u64 + *number_of_partitions as u64 - 1,
                    *number_of_partitions as u64,
                )
            }
            NumberOfPartitions::TwoLayers(layers) => {
                // sum the two layers
                binomial_coefficient(
                    self.number_of_objectives as u64 + layers.boundary_layer as u64 - 1,
                    layers.boundary_layer as u64,
                ) + binomial_coefficient(
                    self.number_of_objectives as u64 + layers.inner_layer as u64 - 1,
                    layers.inner_layer as u64,
                )
            }
        }
    }

    /// Generate the vector of weights of reference points.
    ///
    /// return: `Vec<Vec<f64>>`. The vector of weights of size `self.number_of_points`. Each
    /// nested vector, of size equal to `self.number_of_objectives`, contains the relative
    /// coordinates (between 0 and 1) of the points for each objective.
    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        match &self.number_of_partitions {
            NumberOfPartitions::OneLayer(number_of_partitions) => {
                let mut final_weights: Vec<Vec<f64>> = vec![];
                let mut initial_empty_weight: Vec<usize> = vec![0; self.number_of_objectives];
                // start from first objective
                let obj_index: usize = 0;

                self.recursive_weights(
                    &mut final_weights,
                    &mut initial_empty_weight,
                    *number_of_partitions,
                    *number_of_partitions,
                    obj_index,
                );
                final_weights
            }
            NumberOfPartitions::TwoLayers(layers) => {
                // Create the two layers
                let mut final_weights: Vec<Vec<f64>> = vec![];
                let mut initial_empty_weight: Vec<usize> = vec![0; self.number_of_objectives];
                let obj_index: usize = 0;
                self.recursive_weights(
                    &mut final_weights,
                    &mut initial_empty_weight,
                    layers.boundary_layer,
                    layers.boundary_layer,
                    obj_index,
                );

                let mut inner_points: Vec<Vec<f64>> = vec![];
                let mut initial_empty_weight: Vec<usize> = vec![0; self.number_of_objectives];
                let obj_index: usize = 0;
                self.recursive_weights(
                    &mut inner_points,
                    &mut initial_empty_weight,
                    layers.inner_layer,
                    layers.inner_layer,
                    obj_index,
                );
                // scale the inner layer and then merge it
                let scaling = layers.scaling.unwrap_or(0.5);
                for inner_point in inner_points {
                    let new_points = inner_point
                        .iter()
                        .map(|value| (1.0 / self.number_of_objectives as f64 + value) * scaling)
                        .collect();
                    final_weights.push(new_points);
                }
                final_weights
            }
        }
    }

    /// Calculate the coordinates for each reference point or weight recursively for each objective
    /// and partition index.
    ///
    /// # Arguments
    ///
    /// * `final_weights`: The vector with the final weights.
    /// * `weight`: The vector for a weight or reference point. This must have a size equal to the
    ///    number of objectives.
    /// * `left_partitions`: The number of partition left to process for the objective.
    /// * `number_of_partitions`: The number of total partitions.
    /// * `obj_index`: The objective index being process.
    ///
    /// returns: The vector of weights of size [`self.number_of_points`] is stored in
    /// `final_weights`. Each nested vector, of  size equal to [`self.number_of_objectives`],
    /// contains the relative coordinates (between 0 and 1) for each objective.
    fn recursive_weights(
        &self,
        final_weights: &mut Vec<Vec<f64>>,
        weight: &mut Vec<usize>,
        left_partitions: usize,
        number_of_partitions: usize,
        obj_index: usize,
    ) {
        for k in 0..=left_partitions {
            if obj_index != self.number_of_objectives - 1 {
                // keep processing the left partitions for the next objective
                weight[obj_index] = k;
                self.recursive_weights(
                    final_weights,
                    weight,
                    left_partitions - k,
                    number_of_partitions,
                    obj_index + 1,
                )
            } else {
                // process the last point and update the final weight vector when all objectives
                // have been exhausted
                weight[obj_index] = left_partitions;
                final_weights.push(
                    weight
                        .iter()
                        .map(|v| *v as f64 / number_of_partitions as f64)
                        .collect(),
                );
                break;
            }
        }
    }

    /// Export a vector of reference point as JSON file.
    ///
    /// # Arguments
    ///
    /// * `ref_points`: The reference point vector.
    /// * `file`: The path and file where to save the serialised data.
    ///
    /// returns: `Result<(), OError>`
    /// ```
    pub fn serialise(ref_points: &[Vec<f64>], file: &PathBuf) -> Result<(), OError> {
        #[derive(Serialize)]
        pub struct Points<'a>(pub &'a [Vec<f64>]);
        let p = Points(ref_points);

        let data = serde_json::to_string_pretty(&p.0).map_err(|e| {
            OError::AlgorithmExport(format!(
                "The following error occurred while converting the history struct: {e}"
            ))
        })?;

        fs::write(file, data).map_err(|e| {
            OError::AlgorithmExport(format!(
                "The following error occurred while exporting the history JSON file: {e}",
            ))
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::test_utils::assert_approx_array_eq;
    use crate::utils::reference_points::{binomial_coefficient, DasDarren1998};
    use crate::utils::{NumberOfPartitions, TwoLayerPartitions};

    #[test]
    /// Test the binomial coefficient using results from the Scipy package.
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(6, 4), 15);
        assert_eq!(binomial_coefficient(1, 3), 0);
        assert_eq!(binomial_coefficient(7, 3), 35);
        assert_eq!(binomial_coefficient(100, 2), 4950);
    }

    #[test]
    /// Test the Das & Darren method with 3 objectives and 3 partitions.
    fn test_das_darren_3obj() {
        let m = DasDarren1998::new(3, &NumberOfPartitions::OneLayer(3)).unwrap();
        let weights = m.get_weights();
        let expected_weights = [
            [0.0, 0.0, 1.0],
            [0.0, 0.333, 0.666],
            [0.0, 0.666, 0.333],
            [0.0, 1.0, 0.0],
            [0.333, 0.0, 0.666],
            [0.333, 0.333, 0.333],
            [0.333, 0.666, 0.0],
            [0.666, 0.0, 0.333],
            [0.666, 0.333, 0.0],
            [1.0, 0.0, 0.0],
        ];
        assert_eq!(weights.len() as u64, m.number_of_points());
        assert_eq!(expected_weights.len(), weights.len());

        for (wi, exp_weight_coordinates) in expected_weights.iter().enumerate() {
            assert_approx_array_eq(&weights[wi], exp_weight_coordinates, Some(0.001));
        }
    }

    #[test]
    /// Test the Das & Darren method with 4 objectives and 5 partitions.
    fn test_das_darren_5obj() {
        let m = DasDarren1998::new(4, &NumberOfPartitions::OneLayer(5)).unwrap();
        let weights = m.get_weights();
        let expected_weights = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.2, 0.8],
            [0.0, 0.0, 0.4, 0.6],
            [0.0, 0.0, 0.6, 0.4],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0, 0.8],
            [0.0, 0.2, 0.2, 0.6],
            [0.0, 0.2, 0.4, 0.4],
            [0.0, 0.2, 0.6, 0.2],
            [0.0, 0.2, 0.8, 0.0],
            [0.0, 0.4, 0.0, 0.6],
            [0.0, 0.4, 0.2, 0.4],
            [0.0, 0.4, 0.4, 0.2],
            [0.0, 0.4, 0.6, 0.0],
            [0.0, 0.6, 0.0, 0.4],
            [0.0, 0.6, 0.2, 0.2],
            [0.0, 0.6, 0.4, 0.0],
            [0.0, 0.8, 0.0, 0.2],
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.8],
            [0.2, 0.0, 0.2, 0.6],
            [0.2, 0.0, 0.4, 0.4],
            [0.2, 0.0, 0.6, 0.2],
            [0.2, 0.0, 0.8, 0.0],
            [0.2, 0.2, 0.0, 0.6],
            [0.2, 0.2, 0.2, 0.4],
            [0.2, 0.2, 0.4, 0.2],
            [0.2, 0.2, 0.6, 0.0],
            [0.2, 0.4, 0.0, 0.4],
            [0.2, 0.4, 0.2, 0.2],
            [0.2, 0.4, 0.4, 0.0],
            [0.2, 0.6, 0.0, 0.2],
            [0.2, 0.6, 0.2, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.6],
            [0.4, 0.0, 0.2, 0.4],
            [0.4, 0.0, 0.4, 0.2],
            [0.4, 0.0, 0.6, 0.0],
            [0.4, 0.2, 0.0, 0.4],
            [0.4, 0.2, 0.2, 0.2],
            [0.4, 0.2, 0.4, 0.0],
            [0.4, 0.4, 0.0, 0.2],
            [0.4, 0.4, 0.2, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.4],
            [0.6, 0.0, 0.2, 0.2],
            [0.6, 0.0, 0.4, 0.0],
            [0.6, 0.2, 0.0, 0.2],
            [0.6, 0.2, 0.2, 0.0],
            [0.6, 0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.2],
            [0.8, 0.0, 0.2, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ];
        assert_eq!(weights.len() as u64, m.number_of_points());
        assert_eq!(expected_weights.len(), weights.len());

        for (wi, exp_weight_coordinates) in expected_weights.iter().enumerate() {
            assert_approx_array_eq(&weights[wi], exp_weight_coordinates, None);
        }
    }

    #[test]
    /// test the two layers
    fn test_das_darren_two_layers() {
        let layers = TwoLayerPartitions {
            boundary_layer: 4,
            inner_layer: 3,
            scaling: Some(0.5),
        };
        let m = DasDarren1998::new(3, &NumberOfPartitions::TwoLayers(layers)).unwrap();
        let weights = m.get_weights();
        let expected_weights = [
            [0., 0., 1.],
            [0., 0.25, 0.75],
            [0., 0.5, 0.5],
            [0., 0.75, 0.25],
            [0., 1., 0.],
            [0.25, 0., 0.75],
            [0.25, 0.25, 0.5],
            [0.25, 0.5, 0.25],
            [0.25, 0.75, 0.],
            [0.5, 0., 0.5],
            [0.5, 0.25, 0.25],
            [0.5, 0.5, 0.],
            [0.75, 0., 0.25],
            [0.75, 0.25, 0.],
            [1., 0., 0.],
            [0.16666667, 0.16666667, 0.66666667],
            [0.16666667, 0.33333333, 0.5],
            [0.16666667, 0.5, 0.33333333],
            [0.16666667, 0.66666667, 0.16666667],
            [0.33333333, 0.16666667, 0.5],
            [0.33333333, 0.33333333, 0.33333333],
            [0.33333333, 0.5, 0.16666667],
            [0.5, 0.16666667, 0.33333333],
            [0.5, 0.33333333, 0.16666667],
            [0.66666667, 0.16666667, 0.16666667],
        ];
        assert_eq!(weights.len() as u64, m.number_of_points());
        assert_eq!(expected_weights.len(), weights.len());

        for (wi, exp_weight_coordinates) in expected_weights.iter().enumerate() {
            assert_approx_array_eq(&weights[wi], exp_weight_coordinates, None);
        }
    }
}
