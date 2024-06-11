use std::mem;

use crate::algorithms::fast_non_dominated_sort;
use crate::core::{Individual, Individuals, ObjectiveDirection, OError};
use crate::metrics::hypervolume::{check_args, check_ref_point_coordinate};

/// Calculate the hyper-volume for a two-objective problem by summing the areas rectangle of the
/// rectangles between the Pareto front and the chosen `reference_point`.
#[derive(Debug)]
pub struct HyperVolume2D {
    /// The objective to use. The size of this vector corresponds to th number of problem
    /// objectives (2) and he size of the nested vector corresponds to the number of individuals in
    /// the first Pareto front (of rank 1).
    objective_values: Vec<Vec<f64>>,
    /// The reference point.
    reference_point: Vec<f64>,
}

impl HyperVolume2D {
    /// Calculate the hyper-volume for a two-objective problem. This approach excludes dominated and
    /// unfeasible individuals and then calculates the areas of the rectangles between the convex
    /// Pareto front and the chosen `reference_point`.
    ///
    /// **IMPLEMENTATION NOTES**:
    /// 1) The reference point must dominate the values of all objectives.
    /// 2) For problems with at least one maximised objective, this implementation ensures that the
    /// Pareto front shape is strictly convex and has the same orientation of minimisation problems
    /// by inverting the sign of the objective values to maximise, and the reference point
    /// coordinates.
    /// 3) Dominated and unfeasible solutions are excluded using the NSGA2 [`fast_non_dominated_sort`]
    /// algorithm in order to get the Pareto front (i.e. with non-dominated solutions) to use in
    /// the calculation.
    /// 4) If `individuals` or the resulting Pareto front does not contain more than 2 points, a
    /// zero hyper-volume is returned.
    ///
    /// # Arguments
    ///
    /// * `individuals`: The individuals to use in the calculation. The algorithm will use the
    /// objective vales stored in each individual.
    /// * `reference_point`: The non-dominated reference or anti-optimal point to use in the
    /// calculation. If you are not sure about the point to use, you could pick the worst value of
    /// each objective from the individual's variable using [`crate::metrics::estimate_reference_point`].
    ///
    /// returns: `Result<HyperVolume2D, OError>`
    pub fn new(individuals: &mut [Individual], reference_point: &[f64]) -> Result<Self, OError> {
        let metric_name = "2D Hyper-volume".to_string();

        // check sizes
        check_args(individuals, reference_point)
            .map_err(|e| OError::Metric(metric_name.clone(), e))?;

        if reference_point.len() != 2 {
            return Err(OError::Metric(
                metric_name,
                "This can only be used on a 2-objective problem.".to_string(),
            ));
        }

        // there must be at least two individuals to apply `fast_non_dominated_sort.
        if individuals.len() < 2 {
            return Err(OError::Metric(
                metric_name,
                "At least two individuals are needed to determine the non-dominated solutions"
                    .to_string(),
            ));
        }

        // get non-dominated front
        let problem = individuals[0].problem();
        let mut front_data = fast_non_dominated_sort(individuals, true)?;
        let individuals = mem::take(&mut front_data.fronts[0]);

        // change sign for reference point coordinate. All methods below assume that objectives are minimised.
        let mut ref_point = reference_point.to_vec();
        let mut objective_values: Vec<Vec<f64>> = Vec::new();
        for (obj_idx, (obj_name, obj)) in problem.objectives().iter().enumerate() {
            if obj.direction() == ObjectiveDirection::Maximise {
                ref_point[obj_idx] *= -1.0;
            };
            let objectives = individuals.objective_values(obj_name)?;

            // the reference point must dominate all objectives
            check_ref_point_coordinate(&objectives, obj, reference_point[obj_idx], obj_idx + 1)
                .map_err(|e| OError::Metric(metric_name.clone(), e))?;

            objective_values.push(objectives);
        }

        Ok(Self {
            objective_values,
            reference_point: ref_point,
        })
    }

    /// Calculate the hyper-volume.
    ///
    /// return: `f64`
    pub fn compute(&self) -> f64 {
        // no points in the front
        if self.objective_values.first().is_none() {
            return 0.0;
        }

        // HyperVolume2D::new ensures that there are always 2 objectives and therefore unwrap never
        // panics
        let obj1 = self.objective_values.first().unwrap();
        let obj2 = self.objective_values.last().unwrap();

        // Sort points in descending order by objective 1
        let mut zipped: Vec<_> = obj1.iter().zip(obj2).collect();
        zipped.sort_by(|(a, _), (b, _)| b.total_cmp(a));
        let (obj1_sorted, obj2_sorted): (Vec<f64>, Vec<f64>) = zipped.into_iter().unzip();

        // rectangle heights (last rectangle is between max y point and reference point)
        let mut y_rectangles = obj2_sorted.clone();
        y_rectangles.push(self.reference_point[1]);
        let heights: Vec<f64> = y_rectangles
            .windows(2)
            .map(|y| f64::abs(y[0] - y[1]))
            .collect();

        // get rectangle areas (x is always between objective 1 and reference point)
        obj1_sorted
            .iter()
            .zip(heights)
            .map(|(x_o1, h)| (self.reference_point[0] - x_o1).abs() * h.abs())
            .sum()
    }
}

#[cfg(test)]
/// Test the hyper-volume calculation in 2D. Expected value was manually calculated.
mod test {
    use std::sync::Arc;

    use float_cmp::{approx_eq, assert_approx_eq};

    use crate::core::{
        BoundedNumber, Constraint, Individual, Objective, ObjectiveDirection, Problem,
        RelationalOperator, VariableType, VariableValue,
    };
    use crate::core::utils::{
        dummy_evaluator, individuals_from_obj_values_dummy, individuals_from_obj_values_ztd1,
    };
    use crate::metrics::hypervolume_2d::HyperVolume2D;
    use crate::metrics::test_utils::parse_pagmo_test_data_file;

    #[test]
    /// Test that an error is returned if the reference point does not dominate the objectives
    fn test_ref_point_error() {
        let obj_values = vec![[1.0, 2.0], [0.5, 4.0], [0.0, 6.0]];

        // Minimise both
        let mut individuals = individuals_from_obj_values_dummy(
            &obj_values,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Minimise],
        );

        // x too small
        let ref_point = [0.2, 20.0];
        let hv = HyperVolume2D::new(&mut individuals, &ref_point);
        let err = hv.unwrap_err().to_string();
        assert!(err.contains("The coordinate #1 of the reference point (0.2) must be strictly larger than the maximum value of objective 'obj0'"), "{}", err);
        // y too small
        let ref_point = [20.0, 1.0];
        let hv = HyperVolume2D::new(&mut individuals, &ref_point);
        let err = hv.unwrap_err().to_string();
        assert!(err.contains("The coordinate #2 of the reference point (1) must be strictly larger than the maximum value of objective 'obj1'"), "{}", err);

        // Maximise obj 1 - x too large
        let mut individuals = individuals_from_obj_values_dummy(
            &obj_values,
            &[ObjectiveDirection::Maximise, ObjectiveDirection::Minimise],
        );
        let ref_point = [6.0, 20.0];
        let hv = HyperVolume2D::new(&mut individuals, &ref_point);
        let err = hv.unwrap_err().to_string();
        assert!(err.contains("The coordinate #1 of the reference point (6) must be strictly smaller than the minimum value of objective 'obj0'"), "{}", err);

        // Maximise obj 2 - y too large
        let mut individuals = individuals_from_obj_values_dummy(
            &obj_values,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Maximise],
        );
        let ref_point = [20.0, 19.0];
        let hv = HyperVolume2D::new(&mut individuals, &ref_point);
        let err = hv.unwrap_err().to_string();
        assert!(err.contains("The coordinate #2 of the reference point (19) must be strictly smaller than the minimum value of objective 'obj1'"), "{}", err);
    }

    #[test]
    // All non-dominated solutions with both objectives being minimised.
    fn test_non_dominated_solutions_min_objectives() {
        let ref_point = [10.0, 10.0];
        let obj_values = [vec![1.0, 2.0], vec![0.5, 4.0], vec![0.0, 6.0]];
        let mut ind = individuals_from_obj_values_ztd1(&obj_values);

        let hv = HyperVolume2D::new(&mut ind, &ref_point);
        assert_eq!(hv.unwrap().compute(), 77.0);

        // mirrored ref point - return error
        let ref_point = [-10.0, -10.0];
        let hv = HyperVolume2D::new(&mut ind, &ref_point);
        assert!(hv.unwrap_err().to_string().contains("must dominate all"));
    }

    #[test]
    /// One solution is dominated
    fn test_dominated_solutions() {
        let ref_point = [10.0, 10.0];
        let obj_values = [vec![1.0, 2.0], vec![3.0, 4.0], vec![0.0, 6.0]];
        let mut ind = individuals_from_obj_values_ztd1(&obj_values);

        let hv = HyperVolume2D::new(&mut ind, &ref_point);
        assert_eq!(hv.unwrap().compute(), 76.0);
    }

    #[test]
    /// Two solution is dominated - this return the area of rectangle between ref point and min
    fn test_two_dominated_solutions() {
        let ref_point = [10.0, 10.0];
        let obj_values = [vec![-1.0, 2.0], vec![0.5, 4.0], vec![0.0, 6.0]];
        let mut ind = individuals_from_obj_values_ztd1(&obj_values);

        let hv = HyperVolume2D::new(&mut ind, &ref_point);
        assert_eq!(hv.unwrap().compute(), 88.0);
    }

    #[test]
    /// Two solution is dominated - cannot use fast sorting
    fn test_one_solutions() {
        let ref_point = [10.0, 10.0];
        let obj_values = [vec![-1.0, -2.0]];
        let mut ind = individuals_from_obj_values_ztd1(&obj_values);

        let hv = HyperVolume2D::new(&mut ind, &ref_point);
        assert_eq!(hv.unwrap().compute(), 0.0);
    }

    #[test]
    /// Non-dominated but one is unfeasible which is removed
    fn test_unfeasible_solutions() {
        let ref_point = [10.0, 10.0];
        let obj_values = [vec![1.0, 2.0], vec![3.0, 4.0], vec![0.0, 6.0]];
        let mut objectives = Vec::new();
        for i in 0..obj_values[0].len() {
            objectives.push(Objective::new(
                format!("obj{i}").as_str(),
                ObjectiveDirection::Minimise,
            ));
        }

        let variables = vec![VariableType::Real(
            BoundedNumber::new("X", 0.0, 2.0).unwrap(),
        )];
        let constraints = vec![Constraint::new(
            "x_lower_bound",
            RelationalOperator::GreaterOrEqualTo,
            1.0,
        )];
        let problem = Arc::new(
            Problem::new(objectives, variables, Some(constraints), dummy_evaluator()).unwrap(),
        );

        let mut individuals: Vec<Individual> = Vec::new();
        let x_values = [1.6, 0.2, 1.8]; // 2nd var is not feasible
        for (di, data) in obj_values.iter().enumerate() {
            let mut individual = Individual::new(problem.clone());
            individual
                .update_variable("X", VariableValue::Real(x_values[di]))
                .unwrap();
            for (i, obj_value) in data.iter().enumerate() {
                individual
                    .update_objective(format!("obj{i}").as_str(), *obj_value)
                    .unwrap();
            }
            individuals.push(individual);
        }

        let hv = HyperVolume2D::new(&mut individuals, &ref_point);
        assert_eq!(hv.unwrap().compute(), 76.0);
    }

    #[test]
    /// Maximise objective #1
    fn test_maximise_obj_1() {
        let ref_point = [-10.0, 10.0];
        let obj_values = vec![
            [11.1, 8.1],
            [8.1, 6.1],
            [5.1, 4.1],
            [3.1, 3.1],
            [2.1, 2.1],
            [1.1, 1.1],
            [0.0, 5.1],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &obj_values,
            &[ObjectiveDirection::Maximise, ObjectiveDirection::Minimise],
        );

        let hv = HyperVolume2D::new(&mut individuals, &ref_point);
        assert_approx_eq!(f64, hv.unwrap().compute(), 142.79, ulps = 2);
    }

    #[test]
    /// Maximise objective #2
    fn test_maximise_obj_2() {
        let ref_point = [15.0, -10.0];
        let obj_values = vec![
            [11.1, 8.1],
            [6.1, 6.1],
            [5.1, 5.1],
            [3.1, 3.1],
            [2.1, 2.1],
            [1.1, 1.1],
        ];
        let mut individuals = individuals_from_obj_values_dummy(
            &obj_values,
            &[ObjectiveDirection::Minimise, ObjectiveDirection::Maximise],
        );

        let hv = HyperVolume2D::new(&mut individuals, &ref_point);
        assert_approx_eq!(f64, hv.unwrap().compute(), 215.59, ulps = 2);
    }

    #[test]
    /// Test the `HyperVolume2D` struct using Pagmo c_max_t100_d2_n128 test data.
    /// See https://github.com/esa/pagmo2/tree/master/tests/hypervolume_test_data
    fn test_c_max_t1_d3_n2048() {
        let all_test_data = parse_pagmo_test_data_file::<2>("c_max_t100_d2_n128").unwrap();
        let objective_direction = [ObjectiveDirection::Minimise; 2];

        for (ti, test_data) in all_test_data.iter().enumerate() {
            let mut individuals = individuals_from_obj_values_dummy(
                &test_data.objective_values,
                &objective_direction,
            );
            let hv = HyperVolume2D::new(&mut individuals, &test_data.reference_point).unwrap();

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
}
