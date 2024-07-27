use std::fs::read_to_string;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use float_cmp::approx_eq;

use crate::core::{
    BoundedNumber, Individual, Objective, ObjectiveDirection, Problem, utils, VariableType,
};
use crate::core::VariableValue::Real;

/// Compare two arrays of f64
pub(crate) fn assert_approx_array_eq(
    calculated_values: &[f64],
    expected_values: &[f64],
    epsilon: Option<f64>,
) {
    let epsilon = epsilon.unwrap_or(0.00001);
    for (i, (calculated, expected)) in calculated_values.iter().zip(expected_values).enumerate() {
        if !approx_eq!(f64, *calculated, *expected, epsilon = epsilon) {
            panic!(
                r#"assertion failed on item #{i:?}
                    actual: `{calculated:?}`,
                    expected: `{expected:?}`"#,
            )
        }
    }
}

/// Get the vector values outside a lower and upper bounds.
///
/// # Arguments
///
/// * `vector`: The vector.
/// * `range`: The range.
///
/// returns: `Vec<f64>` The values outside the range.

pub(crate) fn check_value_in_range(vector: &[f64], range: &Range<f64>) -> Vec<f64> {
    vector
        .iter()
        .filter_map(|v| if !range.contains(v) { Some(*v) } else { None })
        .collect()
}

/// Check if a number matches another one, but using ranges. Return the vector items outside
/// `strict_range`, if their number is above `max_outside_strict_range`; otherwise the items outside
/// items `loose_range`. This is used to check whether a value from a genetic algorithm matches an
/// exact value; sometimes an algorithm gets very close to the expected value but the solution
/// is still acceptable within a tolerance.
///
/// # Arguments
///
/// * `vector`: The vector.
/// * `strict_range`: The strict range.
/// * `loose_range`: The loose bound.
/// * `max_outside_strict_range`: The maximum item numbers that can be outside the `strict_range`.
///
/// returns: `(Vec<f64>, Range<f64>, String)` The values outside the range, the breached range
/// bounds and the name of the breached range.

pub(crate) fn check_exact_value(
    vector: &[f64],
    strict_range: &Range<f64>,
    loose_range: &Range<f64>,
    max_outside_strict_range: usize,
) -> (Vec<f64>, Range<f64>, String) {
    if strict_range == loose_range {
        panic!("Bounds are identical");
    }
    let v_outside = check_value_in_range(vector, strict_range);

    if v_outside.len() > max_outside_strict_range {
        (v_outside, strict_range.clone(), "strict".to_string())
    } else {
        let v_loose_outside = check_value_in_range(&v_outside, loose_range);
        (v_loose_outside, loose_range.clone(), "loose".to_string())
    }
}

/// Read a CSV file with objectives or variables
pub(crate) fn read_csv_test_file(
    file_name: &PathBuf,
    skip_first_col: Option<bool>,
) -> Vec<Vec<f64>> {
    let skip_first_col = skip_first_col.unwrap_or(true);
    let mut values: Vec<Vec<f64>> = vec![];
    for (li, line) in read_to_string(file_name)
        .unwrap_or_else(|_| panic!("Cannot find {:?}", file_name))
        .lines()
        .enumerate()
    {
        if li == 0 {
            continue;
        }
        let point = line
            .split(',')
            .skip(if skip_first_col { 1 } else { 0 })
            .map(|v| v.to_string().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
            .unwrap();
        values.push(point);
    }
    values
}

/// Create the individuals for a `N`-objective dummy problem, where `N` is the number of items in
/// the arrays of `objective_values`.
///
/// # Arguments
///
/// * `objective_values`: The objective values to set on the individuals. A number of individuals
/// equal to this vector size will be created.
/// * `objective_direction`: The `N` directions for each objective.
///
/// returns: `Vec<Individual>`

pub(crate) fn individuals_from_obj_values_dummy(
    objective_values: &[Vec<f64>],
    objective_direction: &[ObjectiveDirection],
    variable_values: Option<&[Vec<f64>]>,
) -> Vec<Individual> {
    // check lengths
    if objective_values.first().unwrap().len() != objective_direction.len() {
        panic!("The objective values must match the direction vector length")
    }

    let mut objectives = Vec::new();
    for (i, direction) in objective_direction.iter().enumerate() {
        objectives.push(Objective::new(format!("obj{i}").as_str(), *direction));
    }
    let variables = if let Some(variable_values) = variable_values {
        (0..variable_values.len())
            .map(|i| {
                VariableType::Real(BoundedNumber::new(format!("X{i}").as_str(), 0.0, 2.0).unwrap())
            })
            .collect()
    } else {
        vec![VariableType::Real(
            BoundedNumber::new("X", 0.0, 2.0).unwrap(),
        )]
    };
    let problem =
        Arc::new(Problem::new(objectives, variables, None, utils::dummy_evaluator()).unwrap());

    // create the individuals
    let mut individuals: Vec<Individual> = Vec::new();
    for (ind_idx, data) in objective_values.iter().enumerate() {
        let mut individual = Individual::new(problem.clone());
        for (oi, obj_value) in data.iter().enumerate() {
            individual
                .update_objective(format!("obj{oi}").as_str(), *obj_value)
                .unwrap();
        }
        if let Some(variable_values) = variable_values {
            for (vi, var_value) in variable_values[ind_idx].iter().enumerate() {
                individual
                    .update_variable(format!("X{vi}").as_str(), Real(*var_value))
                    .unwrap();
            }
        }

        individuals.push(individual);
    }

    individuals
}
