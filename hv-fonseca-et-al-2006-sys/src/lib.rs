#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Calculate the hyper-volume using the algorithm proposed by [Fonseca et al. (2006)](http://dx.doi.org/10.1109/CEC.2006.1688440)
/// for a problem with `d` objectives and `n` individuals. The function calls version 4 of the
/// algorithm, therefore its complexity is O(`n^(d-2)*log n`).
///
/// **IMPLEMENTATION NOTES**:
/// 1) The reference point must dominate the values of all objectives. Dominated solutions by the
///    reference point are automatically excluded from the calculation.
/// 2) The program assumes that all objectives are minimised. Maximisation objectives may be
///    multiplied by -1 to convert them to minimisation.
///
/// # Arguments
///
/// * `data`: The vector with the objective values. The size of this vector must correspond to the
///    number of individuals `n` in the population. Each sub-vector must have size `d` equal to the
///    number of problem objectives.
/// * `ref_point`: The reference or anti-optimal point to use in the calculation of length `d`.
///
/// returns: `f64`. The hyper-volume.
///
/// # Examples
///
/// ```
/// use hv_fonseca_et_al_2006_sys::calculate_hv;
/// let data = [vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]];
/// let ref_point = vec![3.0, 3.0, 3.0];
/// let hv = calculate_hv(&data, &ref_point);
/// assert_eq!(hv, 8.0);
/// ```
pub fn calculate_hv(data: &[Vec<f64>], ref_point: &[f64]) -> f64 {
    let total_objectives = data.first().unwrap().len();
    let total_individuals = data.len();
    let mut flatten_data = data.iter().flatten().cloned().collect::<Vec<f64>>();

    // data, nobj, popsize, reference
    unsafe {
        fpli_hv(
            flatten_data.as_mut_ptr(),
            total_objectives as i32,
            total_individuals as i32,
            ref_point.as_ptr(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::calculate_hv;

    #[test]
    fn test_hv3d() {
        let data = [vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]];
        let ref_point = vec![3.0, 3.0, 3.0];
        let hv = calculate_hv(&data, &ref_point);
        assert_eq!(hv, 8.0);
    }
}
