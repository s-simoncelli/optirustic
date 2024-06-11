#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::c_int;

include!("bindings.rs");

/// Calculate the hyper-volume using the algorithm proposed by [While et al. (2012)](http://dx.doi.org/10.1109/TEVC.2010.2077298)
/// for a problem with `d` objectives and `n` individuals. This function assumes that the objectives
/// are minimised and assumes that the `opt` flag is set to `2` in the conditional compilation.
///
/// **IMPLEMENTATION NOTES**:
/// 1) The program assumes that all objectives are minimised (`MAXIMISATION` is set to `false`
/// during conditional compilation ). Maximisation objectives may be multiplied by -1 to convert
/// them to minimisation.
/// 2) The `opt` numeric variable is set to `2` in the conditional compilation of the `wfg.c` file.
/// This means that some optimisations, such as dominated point exclusion when calculating the
/// exclusive hyper-volume or reverting to simple calculation for the 2-dimensional case, are
/// included.
///
/// # Arguments
///
/// * `objective_values`: The vector with the objective values. The size of this vector must
/// correspond to the number of individuals `n` in the population. Each sub-vector must have size
/// `d` equal to the number of problem objectives.
/// * `ref_point`: The reference or anti-optimal point to use in the calculation of length `d`.
///
/// returns: `Result<f64, String>`. The hyper-volume.
///
/// # Examples
///
/// ```
/// use hv_wfg_sys::calculate_hv;
/// let mut ref_point = vec![0.0, 0.0, 0.0];
/// let mut data = [
///     vec![0.598, 0.737, 0.131, 0.916, 6.745],
///     vec![0.263, 0.740, 0.449, 0.753, 6.964],
///     vec![0.109, 8.483, 0.199, 0.302, 8.872],
/// ];
/// println!("{:?}", calculate_hv(&mut data, &mut ref_point));
/// ```
pub fn calculate_hv(
    objective_values: &mut [Vec<f64>],
    ref_point: &mut [f64],
) -> Result<f64, String> {
    match objective_values.first() {
        None => return Err("There are no individuals in the array".to_string()),
        Some(first) => {
            let num_objs = first.len();
            if (0..=1).contains(&num_objs) {
                return Err("This can only be used on a multi-objective problem.".to_string());
            }
            if ref_point.len() != num_objs {
                return Err(format!("The reference point must have {} values", num_objs));
            }
        }
    };

    // collect input arguments

    let mut points: Vec<point> = objective_values
        .iter_mut()
        .map(|objective_value| point {
            objectives: objective_value.as_mut_ptr(),
        })
        .collect();
    let front = front {
        number_of_individuals: points.len() as c_int,
        number_of_objectives: objective_values.first().unwrap().len() as c_int,
        points: points.as_mut_ptr(),
    };
    let ref_point = point {
        objectives: ref_point.as_mut_ptr(),
    };

    Ok(unsafe { calculate_hypervolume(front, ref_point) })
}

#[cfg(test)]
mod test {
    use crate::calculate_hv;

    #[test]
    fn test_wfg() {
        let mut ref_point = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let mut data = vec![
            vec![0.598, 0.737, 0.131, 0.916, 6.745],
            vec![0.263, 0.740, 0.449, 0.753, 6.964],
            vec![0.109, 8.483, 0.199, 0.302, 8.872],
        ];
        let hv = calculate_hv(&mut data, &mut ref_point).unwrap();
        assert_eq!(hv.ceil(), 26758.0);
    }
}
