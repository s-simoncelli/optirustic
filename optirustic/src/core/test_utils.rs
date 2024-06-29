#[cfg(test)]
use std::ops::Range;

#[cfg(test)]
use float_cmp::{approx_eq, F64Margin};

#[cfg(test)]
/// Compare two arrays of f64
pub(crate) fn assert_approx_array_eq(calculated_values: &[f64], expected_values: &[f64]) {
    let margins = F64Margin {
        epsilon: 2.0,
        ulps: (f64::EPSILON * 2.0) as i64,
    };
    for (i, (calculated, expected)) in calculated_values.iter().zip(expected_values).enumerate() {
        if !approx_eq!(f64, *calculated, *expected, margins) {
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
#[cfg(test)]
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
/// returns: `(Vec<f64>, Range<f64>)` The values outside the range in the tuple second item.
#[cfg(test)]
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
