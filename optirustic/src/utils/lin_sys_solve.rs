use nalgebra::{DMatrix, SVD};

use crate::core::utils::all_close;

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
/// * `check_solution`: Whether to check that the found solution is within strict tolerances. When
/// true and the solution is outside the tolerances, this returns an error.
///
/// returns: `Result<Vec<f64>, String>`
///
/// # Examples
///
/// ```
/// use optirustic::utils::solve_linear_system;
/// let a = vec![
///     vec![1.0, 9.0, -5.0],
///     vec![-3.0, -5.0, -5.0],
///     vec![-2.0, -7.0, 1.0],
/// ];
/// let b = vec![-32.0, -10.0, 13.0];
/// let x = solve_linear_system(&a, &b, true).unwrap();
/// println!("{:?}", x); // Some(vec![5.0, -3.0, 2.0])
/// ```
pub fn solve_linear_system(
    a: &[Vec<f64>],
    b: &[f64],
    check_solution: bool,
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

    if check_solution & !all_close(b.data.as_slice(), found_b.data.as_slice(), None, None) {
        return Err("The solution is outside the tolerance limits".to_string());
    }
    Ok(solution.data.as_vec().clone())
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use crate::core::test_utils::assert_approx_array_eq;
    use crate::utils::lin_sys_solve::solve_linear_system;

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
        let x = solve_linear_system(&a, &b, true).unwrap();

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

        let x = solve_linear_system(&x, &y, true).unwrap();
        assert_approx_eq!(f64, x[0], 1.0, epsilon = 0.0001);
        assert_approx_eq!(f64, x[1], -0.95, epsilon = 0.0001);
    }
}
