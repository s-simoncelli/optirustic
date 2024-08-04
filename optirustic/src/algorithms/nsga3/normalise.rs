use log::debug;

use crate::algorithms::nsga3::NORMALISED_OBJECTIVE_KEY;
use crate::algorithms::NSGA3;
use crate::core::{DataValue, Individual, Individuals, OError};
use crate::utils::{solve_linear_system, vector_max, vector_min, LinearSolverTolerance};

/// This implements "Algorithm 2" in the paper which normalises the population members using the
/// adaptive ideal point and the intercepts of the hyper-plane passing through the extreme points
/// and crossing the objective space axis. Steps 8-10 are ignored because [`NSGA3`] implementation
/// directly uses Das and Darren's approach with already-normalised points.
///
/// This procedure:
///  - updates the ideal point. The new coordinates may differ from the original point if any
///    objective, calculated at the current evolution, is lower than the one at the previous
///    evolution.
///  - store the normalised objective in the objective space, with respect to the new ideal point
///    and hyper-plane intercepts, in the "translated_objective" data key in each [`Individual`].
pub(crate) struct Normalise<'a> {
    /// The coordinate of the ideal point from the previous evolution.
    ideal_point: &'a mut Vec<f64>,
    /// The individuals that need normalisation.
    individuals: &'a mut [Individual],
}

/// Calculated points used in the NSGA3 normalisation algorithm.
#[allow(dead_code)]
pub(crate) struct NormalisationPoints {
    /// The extreme points used to calculate the hyper-plane intercepts.
    pub(crate) extreme_points: Vec<Vec<f64>>,
    /// The objective intercepts of the plane.
    pub(crate) intercepts: Vec<f64>,
}

impl<'a> Normalise<'a> {
    /// Build the [`Normalise`] struct.
    ///
    /// # Arguments
    ///
    /// * `ideal_point`: The coordinate of the ideal point  from the previous evolution.
    /// * `individuals`: The individual that needs normalisation.
    ///
    /// returns: `Result<Normalise, OError>`
    pub fn new(
        ideal_point: &'a mut Vec<f64>,
        individuals: &'a mut [Individual],
    ) -> Result<Self, OError> {
        if individuals.is_empty() {
            return Err(OError::AlgorithmRun(
                "NSGA3-Normalise".to_string(),
                "The vector of individuals is empty".to_string(),
            ));
        }

        Ok(Normalise {
            ideal_point,
            individuals,
        })
    }

    /// Normalise the population members using "Algorithm 2" from the paper. Objectives are first
    /// translated with respect to the new ideal point and then scaled using the intercepts of the
    /// linear hyper-plane passing through the extreme points.
    ///
    /// This updates the ideal point and stores the normalised objective in the [`Individual`]'s
    /// data.
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals with the objectives
    ///
    /// returns: `NormalisationPoints`: The points calculated in the normalisation.
    pub(crate) fn calculate(&mut self) -> Result<NormalisationPoints, OError> {
        // Step 2 - calculate the new ideal point (based on paragraph IV-C), as the minimum value
        // for each objective from the start of the algorithm evolution up to the current evolution
        // step.
        let problem = self.individuals.first().unwrap().problem();
        for (j, obj_name) in problem.objective_names().iter().enumerate() {
            let new_min = vector_min(&self.individuals.objective_values(obj_name)?)?;
            // update the point if its coordinate is smaller
            if new_min < self.ideal_point[j] {
                self.ideal_point[j] = new_min;
            }
        }
        debug!("Set ideal point to {:?}", self.ideal_point);

        // Step 3 - Translate the individuals' objectives with respect to the ideal point. This
        // implements the calculation of `f'_j(x)` in section IV-C of the paper.
        for x in self.individuals.iter_mut() {
            let translated_objectives = x
                .get_objective_values()?
                .iter()
                .enumerate()
                .map(|(j, v)| v - self.ideal_point[j])
                .collect();
            debug!("Translated objective to {:?}", translated_objectives);
            x.set_data(
                NORMALISED_OBJECTIVE_KEY,
                DataValue::Vector(translated_objectives),
            );
        }

        // Step 4 - Calculate the vector of extreme points
        let mut extreme_points = vec![];
        for j in 0..problem.number_of_objectives() {
            // Extreme point z_j_max for current objective
            let mut weights = vec![10.0_f64.powi(-6); problem.number_of_objectives()];
            weights[j] = 1.0;

            let mut min_value = f64::INFINITY; // minimum ASF
            let mut ind_index = 0; // index of individual with minimum ASF
            for (x_idx, ind) in self.individuals.iter().enumerate() {
                let f_j = NSGA3::get_normalised_objectives(ind)?;
                let value = self.asf(f_j.as_f64_vec()?, &weights)?;

                if value < min_value {
                    min_value = value;
                    ind_index = x_idx;
                }
            }
            extreme_points.push(
                NSGA3::get_normalised_objectives(&self.individuals[ind_index])?
                    .as_f64_vec()?
                    .clone(),
            );
        }
        debug!("Set extreme points to {:?}", extreme_points);

        // Step 6 - Compute intercepts a_j with the least-square method
        let intercept_result = Self::calculate_plane_intercepts(&extreme_points, None)?;
        let intercepts: Vec<f64> = match intercept_result {
            None => {
                // no solution found or intercepts are too small - get worst (max) for each objective
                let max_points = self.calculate_max_objectives()?;
                debug!("Using maximum points as intercepts {:?}", max_points);
                max_points
            }
            Some(i) => {
                debug!("Found intercepts {:?}", i);
                i
            }
        };

        // Step 7 - Normalize objectives (f_n). The denominator differs from Eq. 5 in the paper
        // because the intercepts are already calculated using the translated objectives. The new
        // values are updated for all individuals.
        for individual in self.individuals.iter_mut() {
            let new_o: Vec<f64> = NSGA3::get_normalised_objectives(individual)?
                .as_f64_vec()?
                .iter()
                .enumerate()
                .map(|(oi, obj_value)| obj_value / intercepts[oi])
                .collect();
            debug!("Normalised objectives to {:?}", new_o);
            individual.set_data(NORMALISED_OBJECTIVE_KEY, DataValue::Vector(new_o));
        }

        Ok(NormalisationPoints {
            extreme_points,
            intercepts,
        })
    }

    /// Use the least square method to calculate the coefficients of the equation of the plane
    /// passing through the vector of `points`. For example, for a 3D system the equation being
    /// used is: $ax + by + cz = 1$. The coefficient vector $x = [a, b, c]$ is found by solving
    /// the linear system $A \cdot x = b$ where `A` is
    ///
    ///          | x_0   y_0   z_0 |
    ///      A = | x_1   y_1   z_1 |
    ///          |       ...       |
    ///          | x_n   y_n   z_n |
    /// `n` the size of `points` and $b = [1, 1, 1]$. The intercepts are then calculated as the
    /// inverse of `x` as $1/x$. For example for the z-axis intercept (with x=0 and y=0), the point
    /// is found by solving $cz = 1$ or $1/x\[2\]$.
    ///
    /// # Arguments
    ///
    /// * `points`: The point coordinates passing through the plane to calculate.
    /// * `tolerance`: The tolerance of the linear solver to accept whether the found solution is
    ///    acceptable.
    ///
    /// returns: `Result<Vec<f64>, OError>`: The $ a_i $ intercept values for each axis (see Fig.2
    /// in the paper) or `None` if the intercepts are close to `0`.
    fn calculate_plane_intercepts(
        points: &[Vec<f64>],
        tolerance: Option<LinearSolverTolerance>,
    ) -> Result<Option<Vec<f64>>, OError> {
        let b = vec![1.0; points.len()];
        let plane_coefficients = solve_linear_system(points, &b, tolerance)
            .map_err(|e| OError::AlgorithmRun("NSGA3-Normalise".to_string(), e))?;
        debug!("Plane coefficients {:?}", plane_coefficients);

        let intercepts: Vec<f64> = plane_coefficients.iter().map(|v| 1.0 / v).collect();

        // check that the intercepts are above the minimum threshold
        if intercepts.iter().all(|v| *v >= 10_f64.powi(-3)) {
            Ok(Some(intercepts))
        } else {
            Ok(None)
        }
    }

    /// Calculate the maximum value for each translated objective.
    ///
    /// return: `Result<Vec<f64>, OError>`
    fn calculate_max_objectives(&self) -> Result<Vec<f64>, OError> {
        let problem = self.individuals.first().unwrap().problem();
        let mut max_points = vec![];
        for j in 0..problem.number_of_objectives() {
            let mut obj_j_values = Vec::new();
            for ind in self.individuals.iter() {
                obj_j_values.push(NSGA3::get_normalised_objectives(ind)?.as_f64_vec()?[j]);
            }
            obj_j_values.push(f64::EPSILON);
            max_points.push(vector_max(&obj_j_values)?);
        }
        debug!("Using maximum points as intercepts {:?}", max_points);
        Ok(max_points)
    }

    /// Calculate the achievement scalarising function with weight vector `w`. This is Eq. 4 in the
    /// paper.
    ///
    /// # Arguments
    ///
    /// * `translated_objective`: The translated objective for an individual. This is f'_j(x).
    /// * `weights`: The weight vector.
    ///
    /// returns: `Result<Vec<f64>, OError>`
    fn asf(&self, translated_objective: &[f64], weights: &Vec<f64>) -> Result<f64, OError> {
        let asf: Vec<f64> = translated_objective
            .iter()
            .zip(weights)
            .map(|(x, w)| x / w)
            .collect();
        vector_max(&asf)
    }
}

#[cfg(test)]
mod test {
    use std::env;
    use std::path::Path;

    use crate::algorithms::nsga3::normalise::Normalise;
    use crate::algorithms::nsga3::NORMALISED_OBJECTIVE_KEY;
    use crate::core::test_utils::{
        assert_approx_array_eq, individuals_from_obj_values_dummy, read_csv_test_file,
    };
    use crate::core::ObjectiveDirection;
    use crate::utils::LinearSolverTolerance;

    #[test]
    /// Test intercepts. Points were generated from numpy from uniform distribution with normal
    /// distributed noise on z coordinates (scale=1). Plane was generated to have slope of -2 in
    /// the x direction and -3 in the y direction.
    fn test_intercepts() {
        let points = vec![
            vec![3.3817863, 0.40604364, -2.2899773],
            vec![4.1741924, 0.92094903, -5.91434001],
            vec![3.42070899, 0.90266942, -3.81063094],
            vec![1.11301849, 0.94849208, 0.17140235],
            vec![9.08303894, 0.74599477, -16.14020622],
            vec![0.98976491, 0.84847939, 0.82864021],
            vec![7.53579489, 0.73723563, -11.72284018],
            vec![6.96274164, 0.59449793, -10.71963907],
            vec![5.60255823, 1.69973452, -12.49841699],
            vec![6.16815342, 0.66601692, -11.63169056],
        ];

        let tol = LinearSolverTolerance {
            relative: 0.01,
            absolute: 0.01,
        };
        let intercepts = Normalise::calculate_plane_intercepts(&points, Some(tol))
            .unwrap()
            .unwrap();
        assert_approx_array_eq(&intercepts, &[3.38096778, 1.61009025, 7.58962871], None);
    }

    #[test]
    /// Test association with DTLZ1 problem from randomly-generated objectives.
    fn test_normalisation_dtlz1() {
        let test_path = Path::new(&env::current_dir().unwrap())
            .join("src")
            .join("algorithms")
            .join("nsga3")
            .join("test_data");

        // Raw objectives
        let obj_file = test_path.join("Normalise_objectives.csv");
        let objectives = read_csv_test_file(&obj_file, None);
        let directions = vec![ObjectiveDirection::Minimise; objectives[0].len()];

        // Expected objectives
        let n_obj_file = test_path.join("Normalise_normalised_objectives.csv");
        let expected_objectives = read_csv_test_file(&n_obj_file, None);
        let expected_ideal_point = [1.346989, 0.391142, 2.169983];
        let expected_extreme_points = [
            [203.1664, 2.5232, 4.4077],
            [29.6926, 60.5373, 29.7112],
            [1.7476, 2.7393, 107.6637],
        ];

        let mut individuals = individuals_from_obj_values_dummy(&objectives, &directions, None);
        let mut ideal_point = vec![f64::INFINITY; 3];
        let mut n = Normalise::new(&mut ideal_point, &mut individuals).unwrap();
        let tmp_points = n.calculate().unwrap();

        // check extreme points
        for (vi, vec) in tmp_points.extreme_points.iter().enumerate() {
            assert_approx_array_eq(vec, expected_extreme_points.get(vi).unwrap(), Some(0.001));
        }

        // check ideal point calculation
        assert_approx_array_eq(&ideal_point, &expected_ideal_point, None);

        // check normalised objectives
        for (ind, expected) in individuals.iter().zip(expected_objectives) {
            let data = ind.get_data(NORMALISED_OBJECTIVE_KEY).unwrap();
            let calculated = data.as_f64_vec().unwrap();
            assert_approx_array_eq(calculated, &expected, None);
        }
    }
}
