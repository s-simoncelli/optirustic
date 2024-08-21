use std::env;
use std::path::PathBuf;

use optirustic::core::OError;
use optirustic::utils::{DasDarren1998, NumberOfPartitions};

/// Generate and plot reference points using the Ds & Darren (1998) method.
fn main() -> Result<(), OError> {
    // Consider the case of a 3D hyperplane with 3 objectives
    let number_of_objectives = 3;
    // Each objective axis is split into 5 gaps of equal size.
    let number_of_partitions = NumberOfPartitions::OneLayer(5);

    let m = DasDarren1998::new(number_of_objectives, &number_of_partitions)?;
    // This returns the coordinates of the reference points between 0 and 1
    println!("Total pints = {:?}", m.number_of_points());

    let weights = m.get_weights();
    println!("Weights = {:?}", weights);

    // Save the serialise data to inspect them
    let out_path = PathBuf::from(&env::current_dir().unwrap())
        .join("examples")
        .join("results")
        .join("ref_points_3obj_5gaps.json");
    DasDarren1998::serialise(&weights, &out_path)?;

    // You can plot the data by running reference_points_1layer_plot.py

    Ok(())
}
