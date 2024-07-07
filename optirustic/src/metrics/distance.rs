use std::iter::Zip;
use std::slice::Iter;

use crate::core::{Individual, OError};
use crate::utils::vector_min;

static DISTANCE_NAME: &str = "Distance";

/// This struct allows calculation of the following distance metrics to assess the performance of a
/// genetic algorithm:
/// 1) Generational Distance (GD)
///
///     $  GD (A, R) =  \frac{1}{| A |} \cdot \[ \sum_{a \in A } min_{r \in R} \quad  d(a, r)^p \]^{1/p} $
///
/// 2) Inverted Generational Distance (IGD)
///
///     $  IGD (A, R) =  GD (R, A) = \frac{1}{| R |} \cdot \[ \sum_{r \in R } min_{a \in A} \quad  d(r, a)^p \]^{1/p} $
///
/// 3) Generational Distance Plus (GD+)
///
///     $  GD_+ (A, R) =  \frac{1}{| A |} \cdot \[ \sum_{a \in A } min_{r \in R} \quad  d^{+}(a, r)^p \]^{1/p} $
///
/// 4) Inverted Generational Distance Plus (IGD+)
///
///     $  IGD_+ (A, R) =  GD (R, A) = \frac{1}{| R |} \cdot \[ \sum_{r \in R } min_{a \in A} \quad  d^{+}(r, a)^p \]^{1/p} $
///
/// 5) Averaged Hausdorff distance ($delta_P$):
///
///     $ \Delta_P = max\[IGD_p(R,A), \quad IGD_p(A, R)\] $
///
///  with $IGD_p(A, R)$ the modified IGD where $ 1/ | A | $ is elevated to $1/p$
///
///
///   $ IGD_{p}(A, R) = \[ \frac{1}{| R |} \cdot \sum_{r \in R } min_{a \in A} \quad  d(r, a)^p \]^{1/p} $
///
///
/// where
/// - $p$ is an exponent set to `1`;
/// - $a$ is an objective point on the hyper-space belonging to the front $A$;
/// - $r$ is the coordinates for a reference point belonging to the reference front $R$. $R$ is either
///   the true Pareto front or a good approximation of it;
/// - $|A|$ the number of points in the front;
/// - $d$ is the distance $ \sqrt{ \sum_{j \in 1 }^{M} (a_k-r_k)^2 } $ with `M` being the objective
///   number;
/// - $d^{+}$ is the distance $ \sqrt{ \sum_{j \in 1 }^{M} \[max(a_k-r_k, 0) \]^2 } $.
///
/// # Definitions
/// [Ishibuchi et al. (2015)](https://ci-labo-omu.github.io/assets/paper/pdf_file/multiobjective/EMO_2015_IGD_Camera-Ready.pdf)
/// provides a good explanation of all the equations and their meaning.
///
/// # Notes
/// - the GD and IGD metrics are not Pareto-compliant and is not a good metric to assess the
///   quality of a Pareto front (i.e. the metrics may give low distance for a non-optional front);
/// - the IGD+ is weakly Pareto compliant.
pub struct Distance<'a> {
    /// The vector of objective values.
    objectives: Vec<Vec<f64>>,
    /// The reference points to use to calculate the distance. This should be either the true Pareto
    /// front or its good approximation.
    reference_front: &'a [Vec<f64>],
}

impl<'a> Distance<'a> {
    /// Create the distance metric. This returns an error if the size of the reference points does
    /// not equal the number of objectives (i.e. the size of the nested objective vectors).
    ///
    /// # Arguments
    ///
    /// * `individuals`: The vector of individuals.
    /// * `reference_front`: The reference points to use to calculate the distance. This should be
    /// either the true Pareto front or its good approximation. The length of each point must be `M`.
    ///
    /// returns: `Distance`
    pub fn new(
        individuals: &'a [Individual],
        reference_front: &'a [Vec<f64>],
    ) -> Result<Self, OError> {
        if individuals.is_empty() {
            return Err(OError::Metric(
                DISTANCE_NAME.to_string(),
                "The vector of individuals is empty".to_string(),
            ));
        }
        if reference_front.is_empty() {
            return Err(OError::Metric(
                DISTANCE_NAME.to_string(),
                "The vector of reference points is empty".to_string(),
            ));
        }
        if reference_front[0].len() != individuals[0].problem().number_of_objectives() {
            return Err(OError::Metric(
                DISTANCE_NAME.to_string(),
                "Each reference point must have a size equal to the number of objectives"
                    .to_string(),
            ));
        }
        let objectives = individuals
            .iter()
            .map(|i| i.get_objective_values())
            .collect::<Result<Vec<Vec<f64>>, OError>>()?;

        Ok(Self {
            objectives,
            reference_front,
        })
    }

    /// Calculate the generational distance (GD).
    ///
    /// # Reference
    /// > David A. Van Veldhuizen, Gary B. Lamont (1998). “Evolutionary Computation and Convergence to
    /// a Pareto Front.” In John R. Koza (ed.), Late Breaking Papers at the Genetic Programming 1998
    /// Conference, 221-228.
    ///
    /// returns: `Result<f64, OError>`
    pub fn generational_distance(&self) -> Result<f64, OError> {
        Self::_generational_distance(
            &self.objectives,
            self.reference_front,
            Distance::squared_distance,
            false,
            Some(1),
        )
    }

    /// Calculate the inverted generational distance (IGD).
    ///
    /// # Reference
    /// > K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using
    /// Reference-Point-Based Non-dominated Sorting Approach, Part I: Solving Problems With Box
    /// Constraints," in IEEE Transactions on Evolutionary Computation, vol. 18, no. 4, pp. 577-601,
    /// Aug. 2014, doi: 10.1109/TEVC.2013.2281535
    ///
    /// returns: `Result<f64, OError>`
    pub fn inverted_generational_distance(&self) -> Result<f64, OError> {
        Self::_generational_distance(
            self.reference_front,
            &self.objectives,
            Distance::squared_distance,
            false,
            Some(1),
        )
    }

    /// Calculate the generational distance plus (GD+).
    ///
    /// returns: `Result<f64, OError>`
    pub fn generational_distance_plus(&self) -> Result<f64, OError> {
        Self::_generational_distance(
            &self.objectives,
            self.reference_front,
            Distance::max_distance,
            false,
            Some(1),
        )
    }

    /// Calculate the inverted generational distance plus (IGD+).
    ///
    /// # Reference
    /// > Hisao Ishibuchi, Hiroyuki Masuda, Yuki Tanigaki, Yusuke Nojima (2015). “Modified Distance
    /// Calculation in Generational Distance and Inverted Generational Distance.” In António
    /// Gaspar-Cunha, Carlos Henggeler Antunes, Carlos A. Coello Coello (eds.), Evolutionary
    /// Multi-criterion Optimization, EMO 2015 Part I, volume 9018 of Lecture Notes in Computer
    /// Science, 110--125. Springer, Heidelberg, Germany.
    ///
    /// returns: `Result<f64, OError>`
    pub fn inverted_generational_distance_plus(&self) -> Result<f64, OError> {
        Self::_generational_distance(
            self.reference_front,
            &self.objectives,
            Distance::max_distance,
            false,
            Some(1),
        )
    }

    /// Calculate the averaged Hausdorff distance ($\Delta_P$).
    ///
    /// # Reference
    /// > Oliver Schütze, X Esquivel, A Lara, Carlos A. Coello Coello (2012). “Using the Averaged
    /// Hausdorff Distance as a Performance Measure in Evolutionary Multiobjective Optimization.”
    /// IEEE Transactions on Evolutionary Computation, 16(4), 504--522.
    ///
    /// returns: `Result<f64, OError>`
    pub fn hausdorff_distance(&self) -> Result<f64, OError> {
        Ok(f64::max(
            Distance::_generational_distance(
                &self.objectives,
                self.reference_front,
                Distance::squared_distance,
                true,
                None,
            )?,
            Distance::_generational_distance(
                self.reference_front,
                &self.objectives,
                Distance::squared_distance,
                true,
                None,
            )?,
        ))
    }

    /// Squared distance between two points used in the GD and IGD metrics.
    ///
    /// # Arguments
    ///
    /// * `iter`: The iterator with the point coordinates.
    ///
    /// returns: `f64`
    fn squared_distance(iter: Zip<Iter<f64>, Iter<f64>>) -> f64 {
        iter.map(|(a_k, r_k)| (a_k - r_k).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Max (plus) distance between two points used in the GD+ and IGD+ metrics.
    ///
    /// # Arguments
    ///
    /// * `iter`: The iterator with the point coordinates.
    ///
    /// returns: `f64`
    fn max_distance(iter: Zip<Iter<f64>, Iter<f64>>) -> f64 {
        iter.map(|(a_k, r_k)| (a_k - r_k).max(0.0).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate the generational distance of `a` as the distance between each point in the set
    /// and the closest point in the `r` set, averaged over the size of `a`.
    ///
    /// # Arguments
    ///
    /// * `a`: The vector of objectives.
    /// * `r`: The reference points to use to calculate the distance.
    /// * `distance_function`: The function to use to calculate the distance between the points.
    /// * `p`: The exponent to use in the calculation. Default to 1.
    /// * `is_hausdorff`: Whether to elevate the inverse of the counter $1/|A|$ to $1/p$. This must
    /// be `true` when calculating the Hausdorff distance, `false` otherwise.
    ///
    /// returns: `Result<f64, OError>`
    fn _generational_distance<F>(
        a: &[Vec<f64>],
        r: &[Vec<f64>],
        distance_function: F,
        is_hausdorff: bool,
        p: Option<u8>,
    ) -> Result<f64, OError>
    where
        F: Fn(Zip<Iter<f64>, Iter<f64>>) -> f64,
    {
        let p = p.unwrap_or(1);

        let distance_sum = a
            .iter()
            .map(|a| {
                // distances from each ref point
                let distances: Vec<f64> = r
                    .iter()
                    .map(|r| distance_function(a.iter().zip(r)).powi(p as i32))
                    .collect();
                vector_min(&distances)
            })
            .sum::<Result<f64, OError>>()?;

        let exponent = 1.0 / (p as f64);
        if is_hausdorff {
            // for Hausdorff distance
            Ok((distance_sum / a.len() as f64).powf(exponent))
        } else {
            // for GD, GD+, IGD and IGD+
            Ok(distance_sum.powf(exponent) / a.len() as f64)
        }
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use float_cmp::assert_approx_eq;

    use crate::core::ObjectiveDirection;
    use crate::core::utils::individuals_from_obj_values_dummy;
    use crate::metrics::Distance;

    /// Test the distance metrics using two sets of objectives. The values below were manually
    /// calculated to test the Rust implementation.
    #[test]
    fn test_distance() {
        let ref_points: [Vec<f64>; 5] = [
            vec![10.1, 0.3],
            vec![1.53, 6.78],
            vec![1.3, 1.3],
            vec![0.3, 10.1],
            vec![9.123, 8.1],
        ];
        let objective_1 = vec![[4.1, 1.1], [0.3, 9.1], [2.54, 4.67]];
        let mut expected_1 = HashMap::new();
        expected_1.insert("gd", 2.048802387);
        expected_1.insert("igd", 3.924499222);
        expected_1.insert("gd+", 0.0);
        expected_1.insert("igd+", 3.130598114);

        let objective_2 = vec![[8.11, 7.1], [5.67, 5.67], [0.45, 9.1]];
        let mut expected_2 = HashMap::new();
        expected_2.insert("gd", 2.218985866);
        expected_2.insert("igd", 3.627049929);
        expected_2.insert("gd+", 0.05);
        expected_2.insert("igd+", 0.882687127);

        let expected = [expected_1, expected_2];
        let directions = [ObjectiveDirection::Minimise; 2];
        for (objective, expected) in [objective_1, objective_2].iter().zip(expected) {
            let individuals = individuals_from_obj_values_dummy(objective, &directions);
            let metric = Distance::new(&individuals, &ref_points).unwrap();

            assert_approx_eq!(
                f64,
                metric.generational_distance().unwrap(),
                *expected.get("gd").unwrap(),
                epsilon = 0.00001
            );
            assert_approx_eq!(
                f64,
                metric.inverted_generational_distance().unwrap(),
                *expected.get("igd").unwrap(),
                epsilon = 0.00001
            );

            assert_approx_eq!(
                f64,
                metric.generational_distance_plus().unwrap(),
                *expected.get("gd+").unwrap(),
                epsilon = 0.00001
            );
            assert_approx_eq!(
                f64,
                metric.inverted_generational_distance_plus().unwrap(),
                *expected.get("igd+").unwrap(),
                epsilon = 0.00001
            );
        }
    }
}
