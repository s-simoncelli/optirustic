use optirustic::core::OError;
use optirustic::utils::{DasDarren1998, NumberOfPartitions, TwoLayerPartitions};

fn main() -> Result<(), OError> {
    // Consider the case of a 3D hyperplane with 3 objectives
    let number_of_objectives = 3;
    // Each objective axis is split into 5 gaps of equal size.
    let number_of_partitions = 5;

    let layers = TwoLayerPartitions {
        // In the first layer points have a gap of 5
        boundary_layer: 5,
        // In the second layer points have a gap of 3
        inner_layer: 4,
        scaling: None,
    };
    let partitions = NumberOfPartitions::TwoLayers(layers);
    let m = DasDarren1998::new(number_of_objectives, &partitions)?;
    // This returns the coordinates of the reference points between 0 and 1
    println!("Total points = {:?}", m.number_of_points());

    let weights = m.get_weights();
    println!("Weights = {:?}", weights);

    // Save the charts of points to inspect them
    m.plot("ref_points_2layers_3obj_5gaps.png")
}
