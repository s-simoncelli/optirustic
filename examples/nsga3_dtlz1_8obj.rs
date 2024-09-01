use std::env;
use std::error::Error;
use std::path::PathBuf;

use log::LevelFilter;

use optirustic::algorithms::{
    Algorithm, MaxGeneration, NSGA3Arg, Nsga3NumberOfIndividuals, StoppingConditionType, NSGA3,
};
use optirustic::core::builtin_problems::DTLZ1Problem;
use optirustic::operators::{PolynomialMutationArgs, SimulatedBinaryCrossoverArgs};
use optirustic::utils::{NumberOfPartitions, TwoLayerPartitions};

/// Solve the DTLZ1 problem from Deb et al. (2013) with 8 objectives.
///
/// Make sure to compile this in release mode to speed up the calculation:
///
/// `cargo run --example nsga3_dtlz1_8obj -p optirustic --release`
fn main() -> Result<(), Box<dyn Error>> {
    // Add log
    env_logger::builder().filter_level(LevelFilter::Info).init();

    // see Table I
    let number_objectives = 8;
    let k: usize = 5;
    let number_variables: usize = number_objectives + k - 1; // M + k - 1 with k = 5 (Section Va)
    let problem = DTLZ1Problem::create(number_variables, number_objectives)?;
    // The number of partitions used in the paper when from section 5
    let number_of_partitions = NumberOfPartitions::TwoLayers(TwoLayerPartitions {
        boundary_layer: 3,
        inner_layer: 2,
        scaling: None,
    });
    // number of individuals - from Table I
    let pop_size: usize = 156;

    // see Table II
    let crossover_operator_options = SimulatedBinaryCrossoverArgs {
        distribution_index: 30.0,
        ..SimulatedBinaryCrossoverArgs::default()
    };
    // eta_m = 20 - probability  1/n_vars
    let mutation_operator_options = PolynomialMutationArgs::default(&problem);

    let args = NSGA3Arg {
        number_of_individuals: Nsga3NumberOfIndividuals::Custom(pop_size),
        number_of_partitions,
        crossover_operator_options: Some(crossover_operator_options),
        mutation_operator_options: Some(mutation_operator_options),
        stopping_condition: StoppingConditionType::MaxGeneration(MaxGeneration(750)),
        parallel: None,
        export_history: None,
        seed: Some(1),
    };

    // Initialise the algorithm
    let mut algo = NSGA3::new(problem, args).unwrap();

    // Run the algorithm
    algo.run()?;

    // Export the last results to a JSON file
    let destination = PathBuf::from(&env::current_dir().unwrap())
        .join("examples")
        .join("results");
    algo.save_to_json(&destination, Some("DTLZ1_8obj"))?;

    Ok(())
}
