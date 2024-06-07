pub use hypervolume::estimate_reference_point;
pub use hypervolume_2d::HyperVolume2D;
pub use hypervolume_fonseca_2006::HyperVolumeFonseca2006;
pub use hypervolume_while_2012::HyperVolumeWhile2012;

pub mod hypervolume;
pub mod hypervolume_2d;
pub mod hypervolume_fonseca_2006;
pub mod hypervolume_while_2012;

#[cfg(test)]
pub(crate) mod test_utils {
    use std::env;
    use std::error::Error;
    use std::fs::read_to_string;
    use std::path::Path;

    #[derive(Debug)]
    /// Pagmo's test data.
    pub(crate) struct TestData<const N: usize> {
        /// The reference point to use in the test.
        pub(crate) reference_point: Vec<f64>,
        /// The objective values to use in the test.
        pub(crate) objective_values: Vec<[f64; N]>,
        /// The expected hyper-volume for the front.
        pub(crate) hyper_volume: f64,
    }

    /// Parse a fle containing Pagmo test data.
    ///
    /// return: `Vec<(Vec<Vec<f64>>, Vec<f64>)>` A vector with a set of tests.
    pub(crate) fn parse_pagmo_test_data_file<const N: usize>(
        filename: &str,
    ) -> Result<Vec<TestData<N>>, Box<dyn Error>> {
        let test_path = Path::new(&env::current_dir().unwrap())
            .join("src")
            .join("metrics")
            .join("test_data");

        // data for one test
        let mut reference_point: Vec<f64> = vec![];
        let mut objective_values: Vec<[f64; N]> = vec![];
        let mut hyper_volume: f64;

        let mut all_test_data: Vec<TestData<N>> = vec![];
        let mut data_line: usize = 0;
        let mut total_points: usize = 0;

        for (li, line) in read_to_string(test_path.join(filename))?
            .lines()
            .enumerate()
        {
            if li == 0 {
                continue;
            }

            // skip number of objectives
            if data_line == 0 {
                data_line += 1;
                continue;
            } else if data_line == 1 {
                total_points = line.to_string().parse::<usize>()?;
                data_line += 1;
                continue;
            } else if data_line == 2 {
                reference_point = line
                    .split(' ')
                    .map(|v| v.to_string().parse::<f64>())
                    .collect::<Result<Vec<f64>, _>>()?;
                data_line += 1;
                continue;
            } else if total_points + 3 > data_line {
                // collect objectives
                let point = line
                    .split(' ')
                    .map(|v| v.to_string().parse::<f64>())
                    .collect::<Result<Vec<f64>, _>>()?;
                let point: [f64; N] = point.try_into().unwrap();
                objective_values.push(point);
                data_line += 1;
                continue;
            } else if total_points + 3 == data_line {
                // get expected hyper-volume
                hyper_volume = line.to_string().parse::<f64>()?;

                // collect test data
                all_test_data.push(TestData {
                    reference_point,
                    objective_values: objective_values.clone(),
                    hyper_volume,
                });

                // reset
                data_line = 0;
                total_points = 0;
                reference_point = vec![];
                objective_values = vec![];
                continue;
            }
        }

        Ok(all_test_data)
    }
}
