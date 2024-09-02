use std::env;
use std::error::Error;
use std::path::PathBuf;

use log::LevelFilter;

use optirustic::algorithms::{
    AdaptiveNSGA3, Algorithm, MaxGenerationValue, NSGA3Arg, Nsga3NumberOfIndividuals,
    StoppingConditionType,
};
use optirustic::core::builtin_problems::DTLZ1Problem;
use optirustic::operators::SimulatedBinaryCrossoverArgs;
use optirustic::utils::NumberOfPartitions;

/// Solve the inverted DTLZ1 problem from Deb et al. (2013) with 3 objectives. This is a problem where the
/// optimal solutions or objectives lie on the hyper-plane passing through the intercept point
/// at 0.5 on each objective axis. This code replicates the first testing problem in Deb et al.
/// (2013).
///
/// Make sure to compile this in release mode to speed up the calculation:
///
/// `cargo run --example nsga3_inverted_dtlz1 -p optirustic --release`
fn main() -> Result<(), Box<dyn Error>> {
    // Add log
    env_logger::builder().filter_level(LevelFilter::Info).init();

    let number_objectives: usize = 3;
    let k: usize = 5;
    // Set the number of variables to use in the DTLZ1 problem
    let number_variables: usize = number_objectives + k - 1;
    // Get the built-in problem
    let problem = DTLZ1Problem::create(number_variables, number_objectives, true)?;

    // Set the number of partitions to create the reference points for the NSGA3 algorithm. This
    // uses one layer of 12 uniform gaps
    let number_of_partitions = NumberOfPartitions::OneLayer(12);

    // Customise the SBX and PM operators like in the paper
    let crossover_operator_options = SimulatedBinaryCrossoverArgs {
        distribution_index: 30.0,
        crossover_probability: 1.0,
        ..SimulatedBinaryCrossoverArgs::default()
    };

    // Set up the adaptive NSGA3 algorithm
    let args = NSGA3Arg {
        // number of individuals from the paper (possibly equal to number of reference points)
        number_of_individuals: Nsga3NumberOfIndividuals::Custom(92),
        number_of_partitions,
        crossover_operator_options: Some(crossover_operator_options),
        mutation_operator_options: None,
        // stop at generation 400
        stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(400)),
        parallel: None,
        export_history: None,
        // to reproduce results
        seed: Some(1),
    };

    // Initialise the algorithm
    let mut algo = AdaptiveNSGA3::new(problem, args).unwrap();

    // Run the algorithm
    algo.run()?;

    // Export the last results to a JSON file
    let destination = PathBuf::from(&env::current_dir().unwrap())
        .join("examples")
        .join("results");

    algo.save_to_json(&destination, Some("DTLZ1_3obj_Adaptive"))?;

    Ok(())
}
