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
    /// The vector of individuals.
    individuals: &'a [Individual],
    /// The reference points to use to calculate the distance. This should be either the true Pareto
    /// front or its good approximation.
    reference_front: Vec<Individual>,
}

impl<'a> Distance<'a> {
    /// Create the distance metric. This returns an error if the size of the reference points does
    /// not equal the number of objectives (i.e. the size of the nested objective vectors).
    ///
    /// # Arguments
    ///
    /// * `individuals`: The vector of individuals.
    /// * `reference_front`: The reference points to use to calculate the distance. This should be
    ///    either the true Pareto front or its good approximation. The length of each point must be `M`.
    ///
    /// returns: `Distance`
    pub fn new(
        individuals: &'a [Individual],
        reference_front: &[Vec<f64>],
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
        let problem = individuals[0].problem();
        if reference_front[0].len() != problem.number_of_objectives() {
            return Err(OError::Metric(
                DISTANCE_NAME.to_string(),
                "Each reference point must have a size equal to the number of objectives"
                    .to_string(),
            ));
        }

        let reference_front = reference_front
            .iter()
            .map(|values| {
                let mut ind = Individual::new(problem.clone());
                problem
                    .objective_names()
                    .iter()
                    .zip(values)
                    .for_each(|(name, value)| ind.update_objective(name, *value).unwrap());
                ind
            })
            .collect();

        Ok(Self {
            individuals,
            reference_front,
        })
    }

    /// Calculate the generational distance (GD).
    ///
    /// # Reference
    /// > David A. Van Veldhuizen, Gary B. Lamont (1998). “Evolutionary Computation and Convergence to
    /// > a Pareto Front.” In John R. Koza (ed.), Late Breaking Papers at the Genetic Programming 1998
    /// > Conference, 221-228.
    ///
    /// returns: `Result<f64, OError>`
    pub fn generational_distance(&self) -> Result<f64, OError> {
        Self::_generational_distance(
            self.individuals,
            &self.reference_front,
            Distance::euclidian_distance,
            false,
            false,
            Some(1),
        )
    }

    /// Calculate the inverted generational distance (IGD).
    ///
    /// # Reference
    /// > K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using
    /// > Reference-Point-Based Non-dominated Sorting Approach, Part I: Solving Problems With Box
    /// > Constraints," in IEEE Transactions on Evolutionary Computation, vol. 18, no. 4, pp. 577-601,
    /// > Aug. 2014, doi: 10.1109/TEVC.2013.2281535
    ///
    /// returns: `Result<f64, OError>`
    pub fn inverted_generational_distance(&self) -> Result<f64, OError> {
        Self::_generational_distance(
            &self.reference_front,
            self.individuals,
            Distance::euclidian_distance,
            true,
            false,
            Some(1),
        )
    }

    /// Calculate the generational distance plus (GD+).
    ///
    /// returns: `Result<f64, OError>`
    pub fn generational_distance_plus(&self) -> Result<f64, OError> {
        Self::_generational_distance(
            self.individuals,
            &self.reference_front,
            Distance::distance_plus,
            false,
            false,
            Some(1),
        )
    }

    /// Calculate the inverted generational distance plus (IGD+).
    ///
    /// # Reference
    /// > Hisao Ishibuchi, Hiroyuki Masuda, Yuki Tanigaki, Yusuke Nojima (2015). “Modified Distance
    /// > Calculation in Generational Distance and Inverted Generational Distance.” In António
    /// > Gaspar-Cunha, Carlos Henggeler Antunes, Carlos A. Coello Coello (eds.), Evolutionary
    /// > Multi-criterion Optimization, EMO 2015 Part I, volume 9018 of Lecture Notes in Computer
    /// > Science, 110--125. Springer, Heidelberg, Germany.
    ///
    /// returns: `Result<f64, OError>`
    pub fn inverted_generational_distance_plus(&self) -> Result<f64, OError> {
        Self::_generational_distance(
            &self.reference_front,
            self.individuals,
            Distance::distance_plus,
            true,
            false,
            Some(1),
        )
    }

    /// Calculate the averaged Hausdorff distance ($\Delta_P$).
    ///
    /// # Reference
    /// > Oliver Schütze, X Esquivel, A Lara, Carlos A. Coello Coello (2012). “Using the Averaged
    /// > ausdorff Distance as a Performance Measure in Evolutionary Multiobjective Optimization.”
    /// > IEEE Transactions on Evolutionary Computation, 16(4), 504--522.
    ///
    /// returns: `Result<f64, OError>`
    pub fn hausdorff_distance(&self) -> Result<f64, OError> {
        Ok(f64::max(
            Distance::_generational_distance(
                self.individuals,
                &self.reference_front,
                Distance::euclidian_distance,
                false,
                true,
                None,
            )?,
            Distance::_generational_distance(
                &self.reference_front,
                self.individuals,
                Distance::euclidian_distance,
                false,
                true,
                None,
            )?,
        ))
    }

    /// Squared distance between two points used in the GD and IGD metrics.
    ///
    /// # Arguments
    ///
    /// * `a`: The first individual.
    /// * `r`: The reference individual.
    /// * `is_inverse`: Whether the distance is for the inverse metric.
    ///
    /// returns: `Result<f64, OError>`
    fn euclidian_distance(
        a: &Individual,
        r: &Individual,
        _is_inverse: bool,
    ) -> Result<f64, OError> {
        Ok(a.get_objective_values()?
            .iter()
            .zip(r.get_objective_values()?)
            .map(|(a_k, r_k)| (a_k - r_k).powi(2))
            .sum::<f64>()
            .sqrt())
    }

    /// Max (plus) distance between two points used in the GD+ and IGD+ metrics.
    ///
    /// # Arguments
    ///
    /// * `a`: The first individual.
    /// * `r`: The reference individual.
    /// * `is_inverse`: Whether the distance is for the inverse metric.
    ///
    /// returns: `Result<f64, OError>`
    fn distance_plus(a: &Individual, r: &Individual, is_inverse: bool) -> Result<f64, OError> {
        let problem = a.problem();
        let distance = problem
            .objective_names()
            .iter()
            .map(|name| {
                // Eq. 18
                let mut delta =
                    a.get_objective_value(name).unwrap() - r.get_objective_value(name).unwrap();
                if is_inverse {
                    delta *= -1.0;
                }
                if !problem.is_objective_minimised(name).unwrap() {
                    delta *= -1.0; // Eq. 19
                }
                delta.max(0.0).powi(2)
            })
            .sum::<f64>()
            .sqrt();

        Ok(distance)
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
    /// * `is_inverse`: Whether the distance is for the inverse metric.
    /// * `is_hausdorff`: Whether to elevate the inverse of the counter $1/|A|$ to $1/p$. This must
    ///    be `true` when calculating the Hausdorff distance, `false` otherwise.
    ///
    /// returns: `Result<f64, OError>`
    fn _generational_distance<F>(
        a: &[Individual],
        r: &[Individual],
        distance_function: F,
        is_inverse: bool,
        is_hausdorff: bool,
        p: Option<u8>,
    ) -> Result<f64, OError>
    where
        F: Fn(&Individual, &Individual, bool) -> Result<f64, OError>,
    {
        let p = p.unwrap_or(1);

        let distance_sum = a
            .iter()
            .map(|a| {
                // distances from each ref point
                let distances: Vec<f64> = r
                    .iter()
                    .map(|r| distance_function(a, r, is_inverse))
                    .collect::<Result<Vec<f64>, OError>>()?;
                // distances
                let distances: Vec<f64> = distances.iter().map(|d| d.powi(p as i32)).collect();
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

    use crate::core::test_utils::individuals_from_obj_values_dummy;
    use crate::core::ObjectiveDirection;
    use crate::metrics::Distance;

    #[test]
    /// Test data from Ishibuchi et al. (2015), Table 4.
    fn test_distance_ishibuchi_et_al_2015() {
        let z = [
            vec![0., 10.],
            vec![1., 6.],
            vec![2., 2.],
            vec![6., 1.],
            vec![10., 0.],
        ];
        let a = [vec![2., 4.], vec![3., 3.], vec![4., 2.]];
        let b = [vec![2., 8.], vec![4., 4.], vec![8., 2.]];
        let directions = [ObjectiveDirection::Minimise; 2];

        // Column I(A)
        let individuals_a = individuals_from_obj_values_dummy(&a, &directions, None);
        let metric = Distance::new(&individuals_a, &z).unwrap();
        assert_approx_eq!(
            f64,
            metric.generational_distance().unwrap(),
            1.805,
            epsilon = 0.001
        );
        assert_approx_eq!(
            f64,
            metric.generational_distance_plus().unwrap(),
            1.138,
            epsilon = 0.001
        );
        assert_approx_eq!(
            f64,
            metric.inverted_generational_distance().unwrap(),
            3.707,
            epsilon = 0.0001
        );
        assert_approx_eq!(
            f64,
            metric.inverted_generational_distance_plus().unwrap(),
            1.483,
            epsilon = 0.001
        );

        // Column I(B)
        let individuals_b = individuals_from_obj_values_dummy(&b, &directions, None);
        let metric = Distance::new(&individuals_b, &z).unwrap();
        assert_approx_eq!(
            f64,
            metric.generational_distance().unwrap(),
            2.434,
            epsilon = 0.001
        );
        assert_approx_eq!(
            f64,
            metric.generational_distance_plus().unwrap(),
            2.276,
            epsilon = 0.001
        );
        assert_approx_eq!(
            f64,
            metric.inverted_generational_distance().unwrap(),
            2.591,
            epsilon = 0.001
        );
        assert_approx_eq!(
            f64,
            metric.inverted_generational_distance_plus().unwrap(),
            2.260,
            epsilon = 0.001
        );
    }

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
        let objective_1 = vec![vec![4.1, 1.1], vec![0.3, 9.1], vec![2.54, 4.67]];
        let mut expected_1 = HashMap::new();
        expected_1.insert("gd", 2.048802387);
        expected_1.insert("igd", 3.924499222);
        expected_1.insert("gd+", 0.0);
        expected_1.insert("igd+", 0.9219999);
        expected_1.insert("hausdorff", 3.9244992221);

        let objective_2 = vec![vec![8.11, 7.1], vec![5.67, 5.67], vec![0.45, 9.1]];
        let mut expected_2 = HashMap::new();
        expected_2.insert("gd", 2.218985866);
        expected_2.insert("igd", 3.627049929);
        expected_2.insert("gd+", 0.05);
        expected_2.insert("igd+", 2.80402265);
        expected_2.insert("hausdorff", 3.6270499);

        let expected = [expected_1, expected_2];
        let directions = [ObjectiveDirection::Minimise; 2];
        for (objective, expected) in [objective_1, objective_2].iter().zip(expected) {
            let individuals = individuals_from_obj_values_dummy(objective, &directions, None);
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
            assert_approx_eq!(
                f64,
                metric.hausdorff_distance().unwrap(),
                *expected.get("hausdorff").unwrap(),
                epsilon = 0.00001
            );
        }
    }
}
