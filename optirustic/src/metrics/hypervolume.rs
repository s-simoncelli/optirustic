use crate::core::{Individual, Individuals, Objective, ObjectiveDirection, OError};
use crate::metrics::{HyperVolumeFonseca2006, HyperVolumeWhile2012};
use crate::metrics::hypervolume_2d::HyperVolume2D;
use crate::utils::{vector_max, vector_min};

/// Calculates a reference point by taking the maximum of each objective (or minimum if the
/// objective is maximised) from the calculated individual's objective values, so that the point will
/// be dominated by all other points. An optional offset for all objectives could also be added or
/// removed to enforce strict dominance (if the objective is minimised the offset is added to the
/// calculated reference point, otherwise it is subtracted).
///
/// # Arguments
///
/// * `individuals`: The individuals to use in the calculation.
/// * `offset`: The offset for each objective to add to the calculated reference point. This must
/// have a size equal to the number of objectives in the problem ([`crate::core::Problem::number_of_objectives`]).
///
/// returns: `Result<Vec<f64>, OError>` The reference point. This returns an error if there are
/// no individuals or the size of the offset does not match [`crate::core::Problem::number_of_objectives`].
pub fn estimate_reference_point(
    individuals: &[Individual],
    offset: Option<Vec<f64>>,
) -> Result<Vec<f64>, OError> {
    let metric_name = "reference_point".to_string();
    if individuals.is_empty() {
        return Err(OError::Metric(
            metric_name,
            "There are no individuals in the array".to_string(),
        ));
    }

    let problem = individuals[0].problem();
    if let Some(ref offset) = offset {
        if offset.len() != problem.number_of_objectives() {
            return Err(OError::Metric(
                metric_name,
                format!(
                    "The offset size ({}) must match the number of problem objectives ({})",
                    offset.len(),
                    problem.number_of_objectives()
                ),
            ));
        }
    }

    let obj_names = problem.objective_names();
    let mut ref_point: Vec<f64> = Vec::new();
    for obj_name in obj_names.iter() {
        let obj_values = individuals.objective_values(obj_name)?;
        // get minimum if objective is minimised
        let coordinate = if problem.is_objective_minimised(obj_name)? {
            vector_max(&obj_values)
        } else {
            vector_min(&obj_values)
        }?;
        ref_point.push(coordinate);
    }

    // add or remove offset
    if let Some(offset) = offset {
        for (idx, name) in obj_names.iter().enumerate() {
            let sign = if problem.is_objective_minimised(name)? {
                1.0
            } else {
                -1.0
            };
            ref_point[idx] += sign * offset[idx];
        }
    }

    Ok(ref_point)
}

/// Check the input arguments of the hyper-volume functions.
///
/// # Arguments
///
/// * `individuals`: The individuals to use in the calculation.
/// * `reference_point`: The reference or anti-optimal point to use in the calculation.
///
/// returns: `Result<(), String>`
pub(crate) fn check_args(
    individuals: &[Individual],
    reference_point: &[f64],
) -> Result<(), String> {
    // Check sizes
    match individuals.first() {
        None => return Err("There are no individuals in the array".to_string()),
        Some(first) => {
            let num_objs = first.problem().number_of_objectives();
            if (0..=1).contains(&num_objs) {
                return Err(
                    "This algorithm can only be used on a multi-objective problem.".to_string(),
                );
            }
        }
    }

    let problem = individuals[0].problem();
    if problem.number_of_objectives() < 2 {
        return Err(
            "The metric can only be calculated on problems with 2 or more objectives".to_string(),
        );
    }
    if reference_point.len() != problem.number_of_objectives() {
        return Err(format!(
                "The number of problem objectives ({}) must match the number of coordinates of the reference point ({})",
                problem.number_of_objectives(), reference_point.len()
            ),
        );
    }

    // check for NaNs
    for (ind_idx, ind) in individuals.iter().enumerate() {
        if ind
            .get_objective_values()
            .map_err(|e| e.to_string())?
            .iter()
            .any(|value| value.is_nan())
        {
            return Err(format!(
                "NaN detected in objective of individual #{}",
                ind_idx
            ));
        }
    }

    Ok(())
}

/// Check that a reference point coordinate dominates all the points of the corresponding
/// objectives. This is essential to ensures that a hyper-volume object is properly calculated.
/// This handles both minimisation and maximisation problem.
///
/// # Arguments
///
/// * `objective_values`: The values of the objective from an individual.
/// * `objective`: The objective being checked.
/// * `ref_point_coordinate`: The coordinate of the reference point.
/// * `coordinate_idx`: The index or position of the coordinate (for example 3 for z-coordinate).
///
/// returns: `Result<(), String>`
pub(crate) fn check_ref_point_coordinate(
    objective_values: &[f64],
    objective: &Objective,
    ref_point_coordinate: f64,
    coordinate_idx: usize,
) -> Result<(), String> {
    // the reference point must dominate all objectives
    let max_obj = vector_max(objective_values).map_err(|e| e.to_string())?;

    if (objective.direction() == ObjectiveDirection::Minimise) & (ref_point_coordinate <= max_obj) {
        return Err(
            format!(
                "The coordinate ({}) of the reference point #{} must be strictly larger than the maximum value of objective '{}' ({}). The reference point must dominate all objectives.",
                ref_point_coordinate ,
                coordinate_idx,
                objective.name(),
                max_obj
        ));
    } else if (objective.direction() == ObjectiveDirection::Maximise)
        & (ref_point_coordinate >= max_obj)
    {
        return Err(
            format!(
                "The coordinate ({}) of the reference point #{} must be strictly smaller than the minimum value of objective '{}' ({}). The reference point must dominate all objectives.",
                ref_point_coordinate,
                coordinate_idx,
                objective.name(),
                -max_obj
        ));
    }
    Ok(())
}

/// Calculate the exact hyper-volume metric for the objectives. Depending on the number of problem
/// objectives `n`, a different method is used to ensure a correct and fast calculation:
///
/// - with `2` objectives: by calculating the rectangle area of each point contributing to the
/// hyper-volume.
/// - with `3` objectives: by using the algorithm proposed by [Fonseca et al. (2006)](http://dx.doi.org/10.1109/CEC.2006.1688440)
/// in [`HyperVolumeFonseca2006`].
/// - with `4` or more objectives:  by using the algorithm proposed by [While et al. (2012)](http://dx.doi.org/10.1109/TEVC.2010.2077298)
/// in [`HyperVolumeWhile2012`].
///
/// # Arguments
///
/// * `individuals`: The individuals to use in the calculation. The algorithm will use the objective
/// vales stored in each individual.
/// * `reference_point`: The reference or anti-optimal point to use in the calculation. If you are
/// not sure about the point to use you could pick the worst value of each objective from the
/// individual's values using [`estimate_reference_point`].
///
/// returns: `Result<f64, OError>`
pub fn hyper_volume(
    individuals: &mut [Individual],
    reference_point: Vec<f64>,
) -> Result<f64, OError> {
    match individuals.first() {
        None => Err(OError::Metric(
            "Hyper-volume".to_string(),
            "There are no individuals in the array".to_string(),
        )),
        Some(first) => {
            let num_objs = first.problem().number_of_objectives();
            if (0..=1).contains(&num_objs) {
                return Err(OError::Metric(
                    "Hyper-volume".to_string(),
                    "This can only be used on a multi-objective problem.".to_string(),
                ));
            }

            let hv_value = match first.problem().number_of_objectives() {
                2 => {
                    let hv = HyperVolume2D::new(individuals, &reference_point)?;
                    hv.compute()
                }
                3 => {
                    let hv = HyperVolumeFonseca2006::new(individuals, &reference_point)?;
                    hv.compute()
                }
                _ => {
                    let mut hv = HyperVolumeWhile2012::new(individuals, &reference_point)?;
                    hv.compute()?
                }
            };
            Ok(hv_value)
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::core::{
        BoundedNumber, Individual, Objective, ObjectiveDirection, Problem, VariableType,
    };
    use crate::core::utils::{dummy_evaluator, individuals_from_obj_values_ztd1};
    use crate::metrics::hypervolume::estimate_reference_point;

    #[test]
    /// Test when the estimate_reference_point function panics
    fn test_worst_point_panic() {
        // no individuals
        let individuals: Vec<Individual> = Vec::new();
        assert!(estimate_reference_point(&individuals, None)
            .unwrap_err()
            .to_string()
            .contains("There are no individuals in the array"));

        // wrong offset size
        let obj_values = vec![vec![-1.0, -2.0], vec![3.0, 4.0], vec![0.0, 6.0]];
        let individuals = individuals_from_obj_values_ztd1(&obj_values);
        let err = estimate_reference_point(&individuals, Some(vec![0.0]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("The offset size (1) must match the number of problem objectives (2)"));
    }

    #[test]
    /// Test when the estimate_reference_point function panics
    fn test_worst_point() {
        // No offset - minimise objectives
        let obj_values = vec![vec![-1.0, -2.0], vec![3.0, 4.0], vec![0.0, 6.0]];
        let individuals = individuals_from_obj_values_ztd1(&obj_values);
        assert_eq!(
            estimate_reference_point(&individuals, None).unwrap(),
            vec![3.0, 6.0]
        );

        // With offset - minimise objectives
        assert_eq!(
            estimate_reference_point(&individuals, Some(vec![1.0, 2.0])).unwrap(),
            vec![4.0, 8.0]
        );

        // Without offset - maximise objectives
        let objectives = vec![
            Objective::new("f1", ObjectiveDirection::Minimise),
            Objective::new("f2", ObjectiveDirection::Maximise),
        ];
        let variables = vec![VariableType::Real(
            BoundedNumber::new("x", 0.0, 1.0).unwrap(),
        )];
        let problem =
            Arc::new(Problem::new(objectives, variables, None, dummy_evaluator()).unwrap());
        let mut individuals = vec![];
        for value in obj_values {
            let mut i = Individual::new(problem.clone());
            i.update_objective("f1", value[0]).unwrap();
            i.update_objective("f2", value[1]).unwrap();
            individuals.push(i);
        }
        assert_eq!(
            estimate_reference_point(&individuals, None).unwrap(),
            vec![3.0, -2.0]
        );

        // With offset - maximise objectives
        assert_eq!(
            estimate_reference_point(&individuals, Some(vec![1.0, 2.0])).unwrap(),
            vec![4.0, -4.0]
        );
    }
}
