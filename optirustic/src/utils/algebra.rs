use nalgebra::{DMatrix, SVD};

use crate::utils::all_close;

/// The tolerance values used to check whether the solver finds acceptable solutions of the linear
/// system. See [`crate::utils::all_close`].
pub struct LinearSolverTolerance {
    pub relative: f64,
    pub absolute: f64,
}

impl Default for LinearSolverTolerance {
    fn default() -> Self {
        Self {
            relative: 1e-05,
            absolute: 1e-08,
        }
    }
}

/// Return the least-squares solution to a linear matrix equation using singular value decomposition
/// (SVD). This solves the linear system `A * x = b`, where `A` is the coefficient matrix of the
/// linear system, `b` the dependent variable and `x` the unknown. `None` is returned if the `A * x`
/// is not close to `b` or the solution does not converge.
///
/// # Arguments
///
/// * `a`: The vector representing the matrix A. This must be a vector whose size is the number of
/// rows in A and each nested vector len is the number of columns in A.
/// * `b`: The column vector b.
/// * `tolerances`: The tolerances to check whether the found solution is acceptable. When the
/// solution is outside the tolerances, this returns an error. When `None`, the solution validity
/// is not checked.
///
/// returns: `Result<Vec<f64>, String>`
///
/// # Examples
///
/// ```
/// use optirustic::utils::{LinearSolverTolerance, solve_linear_system};
/// let a = vec![
///     vec![1.0, 9.0, -5.0],
///     vec![-3.0, -5.0, -5.0],
///     vec![-2.0, -7.0, 1.0],
/// ];
/// let b = vec![-32.0, -10.0, 13.0];
/// let x = solve_linear_system(&a, &b, Some(LinearSolverTolerance::default())).unwrap();
/// println!("{:?}", x); // Some(vec![5.0, -3.0, 2.0])
/// ```
pub fn solve_linear_system(
    a: &[Vec<f64>],
    b: &[f64],
    tolerances: Option<LinearSolverTolerance>,
) -> Result<Vec<f64>, String> {
    // Size check to prevent panic in nalgebra crate
    let num_rows = a.len();
    if num_rows == 0 {
        return Err("The matrix A is empty".to_string());
    }
    let num_cols = a[0].len();
    if num_cols == 0 {
        return Err("The matrix A has no columns".to_string());
    }
    if a.iter().map(|v| v.len()).any(|len| len != num_cols) {
        return Err("All sub-vector in A must have the same number of items".to_string());
    }

    // The number of rows in matrix A and column vector B must match
    let b_num_rows = b.len();
    if b_num_rows != num_rows {
        return Err("The number of rows in A must match the number of rows in B".to_string());
    }

    let flat_a = a.iter().flatten().copied().collect::<Vec<f64>>();
    let a = DMatrix::from_row_slice(num_rows, num_cols, &flat_a);

    let b = DMatrix::from_row_slice(b_num_rows, 1, b);
    let svd = SVD::new(a.clone(), true, true);
    let solution = svd.solve(&b, f64::EPSILON)?;

    // check that the calculated solution is within tolerance
    let found_b = a * &solution;

    if let Some(tolerances) = tolerances {
        if !all_close(
            b.data.as_slice(),
            found_b.data.as_slice(),
            Some(tolerances.relative),
            Some(tolerances.absolute),
        ) {
            return Err("The solution is outside the tolerance limits".to_string());
        }
    }
    Ok(solution.data.as_vec().clone())
}

/// Calculate the dot product between two vectors. The order in which the vectors are given does
/// not matter as the product is commutative. This returns an error if the size of the vectors does
/// not match.
///
/// # Arguments
///
/// * `a`: The first vector.
/// * `b`: The second vector.
///
/// returns: `Result<f64, String>`
pub fn dot_product(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!(
            "The length of vector a ({:?}) must match the length of vector b ({:?})",
            a, b
        ));
    }

    Ok(a.iter().zip(b).map(|(v_a, v_b)| v_a * v_b).sum())
}

/// Get the vector magnitude or length.
///
/// # Arguments
///
/// * `vector`: The vector.
///
/// returns: `Result<f64, String>`
pub fn vector_magnitude(vector: &[f64]) -> Result<f64, String> {
    Ok(dot_product(vector, vector)?.sqrt())
}

/// Calculate the perpendicular distance between a line vector `line` and a `point`. This returns
/// an error if the size of the vectors does not match.
///
/// # Arguments
///
/// * `line`: The reference line.
/// * `point`: The point coordinates.
///
/// returns: `Result<f64, String>`
pub fn perpendicular_distance(line: &[f64], point: &[f64]) -> Result<f64, String> {
    let ref_dir_magnitude = vector_magnitude(line)?;

    // this is a scalar representing the projection of L onto the reference direction
    let projection = dot_product(point, line)? / ref_dir_magnitude;

    let mut distance_vector: Vec<f64> = Vec::with_capacity(point.len());
    for (p, r) in point.iter().zip(line) {
        // projection is multiplied by unit vector (r / ref_dir_magnitude) to get the projection
        // vector parallel to ref_dir.
        let projection_vector = projection * r / ref_dir_magnitude;

        // this is then subtracted from the point vector to get the vector perpendicular to ref_dir
        distance_vector.push(projection_vector - p);
    }

    // get vector length
    vector_magnitude(&distance_vector)
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use crate::core::test_utils::assert_approx_array_eq;
    use crate::utils::algebra::{dot_product, perpendicular_distance, solve_linear_system};
    use crate::utils::LinearSolverTolerance;

    #[test]
    /// Test the lsquare function on a linear system.
    fn test_linear_system() {
        // solve x +9y -5z = -32 / -3x -5y -5z = -10 / -2x - 7y +z = 13
        let a = vec![
            vec![1.0, 9.0, -5.0],
            vec![-3.0, -5.0, -5.0],
            vec![-2.0, -7.0, 1.0],
        ];
        let b = vec![-32.0, -10.0, 13.0];
        let x = solve_linear_system(&a, &b, Some(LinearSolverTolerance::default())).unwrap();

        let expected = vec![5.0, -3.0, 2.0];
        assert_approx_array_eq(&x, &expected);
    }

    #[test]
    /// Test lsquare with linear regression. Example from numpy: <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>
    fn test_linear_regression() {
        let x = vec![
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![3.0, 1.0],
        ];
        let y = vec![-1.0, 0.2, 0.9, 2.1];

        let x = solve_linear_system(&x, &y, Some(LinearSolverTolerance::default())).unwrap();
        assert_approx_eq!(f64, x[0], 1.0, epsilon = 0.0001);
        assert_approx_eq!(f64, x[1], -0.95, epsilon = 0.0001);
    }

    #[test]
    /// Test the dot product function.
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, -5.0, 6.0];
        assert_eq!(12.0, dot_product(&a, &b).unwrap());
    }

    #[test]
    /// Test the perpendicular distance function.
    fn test_perpendicular_distance() {
        let line = vec![1.0, 1.0, 1.0];
        let point = vec![0.0, 0.0, 2.0];
        assert_approx_eq!(
            f64,
            perpendicular_distance(&line, &point).unwrap(),
            1.632993,
            epsilon = 0.0001
        );

        let point = [
            0.027922074966251483,
            0.4628371619519296,
            0.04936679328526684,
        ];
        let lines = [
            [0.08333333333333333, 0.25, 0.6666666666666666],
            [0.08333333333333333, 0.3333333333333333, 0.5833333333333334],
            [0.08333333333333333, 0.75, 0.16666666666666666],
        ];
        let expected_distance = [0.41604855196117385, 0.3774074655777061, 0.05670308534505434];
        let distances = lines
            .iter()
            .map(|l| perpendicular_distance(l, &point).unwrap())
            .collect::<Vec<f64>>();
        assert_approx_array_eq(&distances, &expected_distance);
    }
}
