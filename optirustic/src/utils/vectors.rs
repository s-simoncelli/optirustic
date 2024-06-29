use crate::core::OError;

/// Define the sort type
#[derive(PartialEq)]
pub enum Sort {
    /// Sort values in ascending order
    Ascending,
    /// Sort values in descending order
    Descending,
}

/// Returns the indices that would sort an array in ascending order.
///
/// # Arguments
///
/// * `data`: The vector to sort.
/// * `sort_type`: Specify whether to sort in ascending or descending order.
///
/// returns: `Vec<usize>`. The vector with the indices.
pub fn argsort(data: &[f64], sort_type: Sort) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| data[*a].total_cmp(&data[*b]));

    if sort_type == Sort::Descending {
        indices.reverse();
    }
    indices
}

/// Calculate the vector minimum value.
///
/// # Arguments
///
/// * `v`: The vector.
///
/// returns: `Result<f64, OError>`
///
/// # Examples
pub fn vector_min(v: &[f64]) -> Result<f64, OError> {
    Ok(*v
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .ok_or(OError::Generic(
            "Cannot calculate vector min value".to_string(),
        ))?)
}

/// Calculate the vector maximum value.
///
/// # Arguments
///
/// * `v`: The vector.
///
/// returns: `Result<f64, OError>`
pub fn vector_max(v: &[f64]) -> Result<f64, OError> {
    Ok(*v
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(OError::Generic(
            "Cannot calculate vector max value".to_string(),
        ))?)
}

/// Returns `true` if two arrays are element-wise equal within a tolerance. This behaves as the
///numpy implementation at <https://numpy.org/doc/stable/reference/generated/numpy.allclose.html>.
///
/// # Arguments
///
/// * `a`: First vector to compare.
/// * `b`: Second vector to compare.
/// * `r_tol`: The relative tolerance parameter
/// * `a_tol`: The absolute tolerance parameter
///
/// returns: `bool`
pub fn all_close(a: &[f64], b: &[f64], r_tol: Option<f64>, a_tol: Option<f64>) -> bool {
    let r_tol = r_tol.unwrap_or(1e-05);
    let a_tol = a_tol.unwrap_or(1e-08);

    a.iter()
        .zip(b)
        .any(|(v1, v2)| (v1 - v2).abs() <= (a_tol + r_tol * v2.abs()))
}

#[cfg(test)]
mod test {
    use crate::utils::argsort;
    use crate::utils::vectors::Sort;

    #[test]
    fn test_argsort() {
        let vec = vec![99.0, 11.0, 456.2, 19.0, 0.5];

        assert_eq!(argsort(&vec, Sort::Ascending), vec![4, 1, 3, 0, 2]);
        assert_eq!(argsort(&vec, Sort::Descending), vec![2, 0, 3, 1, 4]);
    }
}
