#[cfg(feature = "plot")]
use std::error::Error;

#[cfg(feature = "plot")]
use plotters::prelude::*;

#[cfg(feature = "plot")]
use crate::core::OError;

/// Calculate the binomial coefficient. This gives the number of `k`-subsets possible out of a
/// set of `n` distinct items. See <https://mathworld.wolfram.com/BinomialCoefficient.html>. Code
/// adapted from <https://blog.plover.com/math/choose.html>.
///
/// # Arguments
///
/// * `n`: The number of possibilities.
/// * `k`: The number of outcomes.
///
/// returns: `u64`
fn binomial_coefficient(mut n: u64, k: u64) -> u64 {
    let mut r: u64 = 1;
    if k > n {
        0
    } else {
        for d in 1..=k {
            r *= n;
            n -= 1;
            r /= d;
        }
        r
    }
}

/// Derive the reference points or weights using the methodology suggested in Section 5.2 in the
/// Das & Dennis (1998) paper:
///
/// > Indraneel Das and J. E. Dennis. Normal-Boundary Intersection: A New Method for Generating the
/// > Pareto Surface in Nonlinear Multicriteria Optimization Problems. SIAM Journal on Optimization.
/// > 1998 8:3, 631-657. <https://doi.org/10.1137/S1052623496307510>
///
/// # Example
/// ```rust
#[doc = include_str!("../../examples/reference_points.rs")]
/// ```
pub struct DasDarren1998 {
    /// The number of problem objectives.
    number_of_objectives: usize,
    /// The number of uniform gaps between two consecutive points along all objective axis on the
    /// hyperplane.
    number_of_partitions: usize,
}

impl DasDarren1998 {
    /// Initialise the Das & Darren approach to calculate reference points or weights.
    ///
    /// # Arguments
    ///
    /// * `number_of_objectives`: The number of problem objectives.
    /// * `number_of_partitions`: The number of uniform gaps between two consecutive points along
    /// all objective axis on the hyperplane.
    ///
    /// returns: `DasDarren1998`
    pub fn new(number_of_objectives: usize, number_of_partitions: usize) -> Self {
        DasDarren1998 {
            number_of_objectives,
            number_of_partitions,
        }
    }

    /// Determine the number of reference points on the `self::number_of_objectives`-dimensional
    /// unit simplex with `self.number_of_partitions` gaps from Section 5.2 of the
    /// [Das & Dennis's paper](https://doi.org/10.1137/S1052623496307510).
    ///
    /// returns: `u64`. The number of reference points.
    pub fn number_of_points(&self) -> u64 {
        // Binomial coefficient of M + p - 1 and p, where M = self.number_of_objectives and
        // p = self.number_of_partitions
        binomial_coefficient(
            self.number_of_objectives as u64 + self.number_of_partitions as u64 - 1,
            self.number_of_partitions as u64,
        )
    }

    /// Generate the vector of weights of reference points.
    ///
    /// return: `Vec<Vec<f64>>`. The vector of weights of size `self.number_of_points`. Each
    /// nested vector, of size equal to `self.number_of_objectives`, contains the relative
    /// coordinates (between 0 and 1) of the points for each objective.
    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        let mut final_weights: Vec<Vec<f64>> = vec![];
        let mut initial_empty_weight: Vec<usize> = vec![0; self.number_of_objectives];
        // start from first objective
        let obj_index: usize = 0;
        self.recursive_weights(
            &mut final_weights,
            &mut initial_empty_weight,
            self.number_of_partitions,
            obj_index,
        );
        final_weights
    }

    /// Calculate the coordinates for each reference point or weight recursively for each objective
    /// and partition index.
    ///
    /// # Arguments
    ///
    /// * `final_weights`: The vector with the final weights.
    /// * `weight`: The vector for a weight or reference point. This must have a size equal to the
    /// number of objectives.
    /// * `left_partitions`: The number of partition left to process for the objective.
    /// * `obj_index`: The objective index being process.
    ///
    /// returns: The vector of weights of size [`self.number_of_points`] is stored in
    /// `final_weights`. Each nested vector, of  size equal to [`self.number_of_objectives`],
    /// contains the relative coordinates (between 0 and 1) for each objective.
    fn recursive_weights(
        &self,
        final_weights: &mut Vec<Vec<f64>>,
        weight: &mut Vec<usize>,
        left_partitions: usize,
        obj_index: usize,
    ) {
        for k in 0..=left_partitions {
            if obj_index != self.number_of_objectives - 1 {
                // keep processing the left partitions for the next objective
                weight[obj_index] = k;
                self.recursive_weights(final_weights, weight, left_partitions - k, obj_index + 1)
            } else {
                // process the last point and update the final weight vector when all objectives
                // have been exhausted
                weight[obj_index] = left_partitions;
                final_weights.push(
                    weight
                        .iter()
                        .map(|v| *v as f64 / self.number_of_partitions as f64)
                        .collect(),
                );
                break;
            }
        }
    }

    /// Generate and save a chart with the reference points. This is only available for problems with
    /// 2 or 3 objectives.
    ///
    /// # Arguments
    ///
    /// * `file_name`: The file path where to save the chart.
    ///
    /// returns: `Result<(), OError>`
    #[cfg(feature = "plot")]
    pub fn plot(&self, file_name: &str) -> Result<(), OError> {
        if self.number_of_objectives == 2 {
            self.plot_2d(file_name)
                .map_err(|e| OError::Generic(e.to_string()))
        } else if self.number_of_objectives == 3 {
            self.plot_3d(file_name)
                .map_err(|e| OError::Generic(e.to_string()))
        } else {
            return Err(OError::Generic(
                "Plotting is available when the number of objective is either 2 or 3".to_string(),
            ));
        }
    }

    /// Generate and save a 2D chart with the reference points.
    ///
    /// # Arguments
    ///
    /// * `file_name`: The file path where to save the chart.
    ///
    /// returns: `Result<(), Box<dyn Error>>`
    #[cfg(feature = "plot")]
    fn plot_2d(&self, file_name: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(65)
            .y_label_area_size(65)
            .margin_top(5)
            .margin_left(10)
            .margin_right(30)
            .margin_bottom(5)
            .caption(
                "Reference points - Das & Darren (2019)",
                ("sans-serif", 30.0),
            )
            .build_cartesian_2d(0f64..1.2f64, 0f64..1.2f64)?;

        chart
            .configure_mesh()
            .bold_line_style(WHITE.mix(0.3))
            .y_desc("Objective #2")
            .x_desc("Objective #1")
            .axis_desc_style(("sans-serif", 25, &BLACK))
            .label_style(("sans-serif", 20, &BLACK))
            .draw()?;

        chart.draw_series(self.get_weights().iter().map(|p| {
            Circle::new(
                (p[0], p[1]),
                5,
                ShapeStyle {
                    color: Palette99::pick(1).to_rgba(),
                    filled: true,
                    stroke_width: 1,
                },
            )
        }))?;

        root.present()?;
        Ok(())
    }

    /// Generate and save a 3D chart with the reference points.
    ///
    /// # Arguments
    ///
    /// * `file_name`: The file path where to save the chart.
    ///
    /// returns: `Result<(), Box<dyn Error>>`
    #[cfg(feature = "plot")]
    fn plot_3d(&self, file_name: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(65)
            .y_label_area_size(65)
            .margin_top(5)
            .margin_left(10)
            .margin_right(30)
            .margin_bottom(5)
            .caption(
                "Reference points - Das & Darren (2019)",
                ("sans-serif", 30.0),
            )
            .build_cartesian_3d(0f64..1.2f64, 0f64..1.2f64, 0f64..1.2f64)?;

        chart.with_projection(|mut pb| {
            pb.yaw = 0.5;
            pb.into_matrix()
        });

        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .max_light_lines(3)
            .draw()?;

        chart.draw_series(self.get_weights().iter().map(|p| {
            Circle::new(
                (p[0], p[1], p[2]),
                5,
                ShapeStyle {
                    color: Palette99::pick(1).to_rgba(),
                    filled: true,
                    stroke_width: 1,
                },
            )
        }))?;

        root.present()?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::test_utils::assert_approx_array_eq;
    use crate::utils::reference_points::{binomial_coefficient, DasDarren1998};

    #[test]
    /// Test the binomial coefficient using results from the Scipy package.
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(6, 4), 15);
        assert_eq!(binomial_coefficient(1, 3), 0);
        assert_eq!(binomial_coefficient(7, 3), 35);
        assert_eq!(binomial_coefficient(100, 2), 4950);
    }

    #[test]
    /// Test the Das & Darren method with 3 objectives and 3 partitions.
    fn test_das_darren_3obj() {
        let m = DasDarren1998::new(3, 3);
        let weights = m.get_weights();
        let expected_weights = [
            [0.0, 0.0, 1.0],
            [0.0, 0.333, 0.666],
            [0.0, 0.666, 0.333],
            [0.0, 1.0, 0.0],
            [0.333, 0.0, 0.666],
            [0.333, 0.333, 0.333],
            [0.333, 0.666, 0.0],
            [0.666, 0.0, 0.333],
            [0.666, 0.333, 0.0],
            [1.0, 0.0, 0.0],
        ];
        assert_eq!(weights.len() as u64, m.number_of_points());

        for (wi, exp_weight_coordinates) in expected_weights.iter().enumerate() {
            assert_approx_array_eq(&weights[wi], exp_weight_coordinates, None);
        }
    }

    #[test]
    /// Test the Das & Darren method with 4 objectives and 5 partitions.
    fn test_das_darren_5obj() {
        let m = DasDarren1998::new(4, 5);
        let weights = m.get_weights();
        let expected_weights = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.2, 0.8],
            [0.0, 0.0, 0.4, 0.6],
            [0.0, 0.0, 0.6, 0.4],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0, 0.8],
            [0.0, 0.2, 0.2, 0.6],
            [0.0, 0.2, 0.4, 0.4],
            [0.0, 0.2, 0.6, 0.2],
            [0.0, 0.2, 0.8, 0.0],
            [0.0, 0.4, 0.0, 0.6],
            [0.0, 0.4, 0.2, 0.4],
            [0.0, 0.4, 0.4, 0.2],
            [0.0, 0.4, 0.6, 0.0],
            [0.0, 0.6, 0.0, 0.4],
            [0.0, 0.6, 0.2, 0.2],
            [0.0, 0.6, 0.4, 0.0],
            [0.0, 0.8, 0.0, 0.2],
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.8],
            [0.2, 0.0, 0.2, 0.6],
            [0.2, 0.0, 0.4, 0.4],
            [0.2, 0.0, 0.6, 0.2],
            [0.2, 0.0, 0.8, 0.0],
            [0.2, 0.2, 0.0, 0.6],
            [0.2, 0.2, 0.2, 0.4],
            [0.2, 0.2, 0.4, 0.2],
            [0.2, 0.2, 0.6, 0.0],
            [0.2, 0.4, 0.0, 0.4],
            [0.2, 0.4, 0.2, 0.2],
            [0.2, 0.4, 0.4, 0.0],
            [0.2, 0.6, 0.0, 0.2],
            [0.2, 0.6, 0.2, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.6],
            [0.4, 0.0, 0.2, 0.4],
            [0.4, 0.0, 0.4, 0.2],
            [0.4, 0.0, 0.6, 0.0],
            [0.4, 0.2, 0.0, 0.4],
            [0.4, 0.2, 0.2, 0.2],
            [0.4, 0.2, 0.4, 0.0],
            [0.4, 0.4, 0.0, 0.2],
            [0.4, 0.4, 0.2, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.4],
            [0.6, 0.0, 0.2, 0.2],
            [0.6, 0.0, 0.4, 0.0],
            [0.6, 0.2, 0.0, 0.2],
            [0.6, 0.2, 0.2, 0.0],
            [0.6, 0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.2],
            [0.8, 0.0, 0.2, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ];
        assert_eq!(weights.len() as u64, m.number_of_points());

        for (wi, exp_weight_coordinates) in expected_weights.iter().enumerate() {
            assert_approx_array_eq(&weights[wi], exp_weight_coordinates, None);
        }
    }
}
