use std::mem;

use log::{debug, warn};

use hv_fonseca_et_al_2006_sys::calculate_hv;

use crate::core::{Individual, Individuals, OError};
use crate::metrics::hypervolume::{check_args, check_ref_point_coordinate};
use crate::utils::fast_non_dominated_sort;

/// Calculate the hyper-volume using the algorithm proposed by [Fonseca et al. (2006)](http://dx.doi.org/10.1109/CEC.2006.1688440)
/// for a problem with `d` objectives and `n` individuals. The function calls version 4 of the
/// algorithm, therefore its complexity is O(`n^(d-2)*log n`).
///
/// **IMPLEMENTATION NOTES**:
/// 1) Points dominated by the reference point are removed from the calculation.
/// 2) Dominated and unfeasible solutions are excluded using the NSGA2 [`crate::utils::fast_non_dominated_sort()`]
/// algorithm in order to get the Pareto front. As assumed in the paper, non-dominated points do
/// not contribute do the metric.
/// 3) The coordinates of maximised objectives of the reference point are multiplied by -1 as the
/// algorithm assumes all objectives are maximised.
#[derive(Debug)]
pub struct HyperVolumeFonseca2006 {
    /// The individuals to use. The size of this vector corresponds to the individual size and the
    /// size of the nested vector corresponds to the number of problem objectives.
    individuals: Vec<Vec<f64>>,
    /// The reference point.
    reference_point: Vec<f64>,
}

impl HyperVolumeFonseca2006 {
    /// Calculate the hyper-volume using the algorithm proposed by [Fonseca et al. (2006)](http://dx.doi.org/10.1109/CEC.2006.1688440)
    /// for a problem with `d` objectives and `n` individuals. The function calls version 4 of the
    /// algorithm, therefore its complexity is O(`n^(d-2)*log n`).
    ///
    /// # Arguments
    ///
    /// * `individuals`: The list of individuals.
    /// * `reference_point`: The reference point.
    ///
    /// returns: `Result<HyperVolumeFonseca2006, OError>`
    pub fn new(individuals: &mut [Individual], reference_point: &[f64]) -> Result<Self, OError> {
        let metric_name = "Hyper-volume Fonseca et al. (2006)".to_string();
        // check sizes
        check_args(individuals, reference_point)
            .map_err(|e| OError::Metric(metric_name.clone(), e))?;

        if reference_point.len() != 3 {
            return Err(OError::Metric(
                metric_name,
                "This can only be used on a 3-objective problem.".to_string(),
            ));
        }

        // the reference point must dominate all objectives
        let problem = individuals[0].problem();
        for (obj_idx, (obj_name, obj)) in problem.objectives().iter().enumerate() {
            check_ref_point_coordinate(
                &individuals.objective_values(obj_name)?,
                obj,
                reference_point[obj_idx],
                obj_idx + 1,
            )
            .map_err(|e| OError::Metric(metric_name.clone(), e))?;
        }

        // get non-dominated front with feasible solutions only
        let num_individuals = individuals.len();
        let mut front_data = fast_non_dominated_sort(individuals, true)?;
        let individuals = mem::take(&mut front_data.fronts[0]);

        if num_individuals != individuals.len() {
            warn!("{} individuals were removed from the given data because they are dominated by all the other points", num_individuals - individuals.len());
        }

        // Collect objective values - invert the sign of minimised objectives
        let objective_values = individuals
            .iter()
            .map(|ind| ind.get_objective_values())
            .collect::<Result<Vec<Vec<f64>>, _>>()?;

        // flip sign of maximised coordinates for the reference point
        let mut ref_point = reference_point.to_vec();
        for (obj_idx, obj_name) in problem.objective_names().iter().enumerate() {
            if !problem.is_objective_minimised(obj_name)? {
                ref_point[obj_idx] *= -1.0;
            }
        }

        debug!("Using non-dominated front {:?}", objective_values);
        debug!("Reference point is {:?}", ref_point);

        Ok(Self {
            individuals: objective_values,
            reference_point: ref_point,
        })
    }

    /// Calculate the hyper-volume.
    ///
    /// return: `f64`
    pub fn compute(&self) -> f64 {
        calculate_hv(&self.individuals, &self.reference_point)
    }
}

#[cfg(test)]
mod test {
    use float_cmp::approx_eq;

    use crate::core::ObjectiveDirection;
    use crate::core::utils::individuals_from_obj_values_dummy;
    use crate::metrics::HyperVolumeFonseca2006;
    use crate::metrics::test_utils::parse_pagmo_test_data_file;

    #[test]
    /// Reference point must be strictly larger than any objective
    fn test_wrong_ref_point() {
        let objective_values = vec![[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]];
        let objective_direction = [ObjectiveDirection::Minimise; 3];

        let mut individuals =
            individuals_from_obj_values_dummy(&objective_values, &objective_direction);
        let ref_point = vec![2.0, 2.0, 2.0];
        let hv = HyperVolumeFonseca2006::new(&mut individuals, &ref_point);
        assert!(hv
            .unwrap_err()
            .to_string()
            .contains("The coordinate (2) of the reference point #1 must be strictly larger"));
    }

    #[test]
    /// Test avery simple front
    fn test_simple_front() {
        let objective_values = vec![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let objective_direction = [ObjectiveDirection::Minimise; 3];
        let ref_point = [3.0, 3.0, 3.0];

        let mut individuals =
            individuals_from_obj_values_dummy(&objective_values, &objective_direction.clone());
        let hv = HyperVolumeFonseca2006::new(&mut individuals, &ref_point).unwrap();
        assert_eq!(hv.compute(), 8.0);
    }

    /// Run a test using a Pagmo file.
    ///
    /// # Arguments
    ///
    /// * `file`: The file name in the `test_data` folder.
    ///
    /// returns: ()
    pub(crate) fn assert_test_file<const N: usize>(file: &str) {
        let all_test_data = parse_pagmo_test_data_file::<N>(file).unwrap();
        let objective_direction = [ObjectiveDirection::Minimise; N];

        for (ti, test_data) in all_test_data.iter().enumerate() {
            let mut individuals = individuals_from_obj_values_dummy(
                &test_data.objective_values,
                &objective_direction,
            );
            let hv =
                HyperVolumeFonseca2006::new(&mut individuals, &test_data.reference_point).unwrap();

            let calculated = hv.compute();
            let expected = test_data.hyper_volume;
            if !approx_eq!(f64, calculated, expected, epsilon = 0.001) {
                panic!(
                    r#"assertion failed for test #{}: `(left approx_eq right)` left: `{:?}`, right: `{:?}`"#,
                    ti + 1,
                    calculated,
                    expected,
                )
            }
        }
    }
    #[test]
    /// Test the `HyperVolumeFonseca2006` struct using Pagmo c_max_t100_d3_n128 test data.
    /// See https://github.com/esa/pagmo2/tree/master/tests/hypervolume_test_data
    fn test_c_max_t100_d3_n128() {
        assert_test_file::<3>("c_max_t100_d3_n128");
    }

    #[test]
    /// Test the `HyperVolumeFonseca2006` struct using Pagmo c_max_t1_d3_n2048 test data.
    /// See https://github.com/esa/pagmo2/tree/master/tests/hypervolume_test_data
    fn test_c_max_t1_d3_n2048() {
        assert_test_file::<3>("c_max_t1_d3_n2048");
    }
}
