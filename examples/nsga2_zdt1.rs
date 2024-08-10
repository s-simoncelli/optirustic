use std::env;
use std::error::Error;
use std::path::PathBuf;

use log::LevelFilter;

use optirustic::algorithms::{Algorithm, MaxGeneration, NSGA2Arg, StoppingConditionType, NSGA2};
use optirustic::core::builtin_problems::ZTD1Problem;

/// Solve the ZDT1 problem (SCH) where the following 2 objectives are minimised:
/// - `f_1(x) = x_1`
/// - `f-2(x) = g(x) * [ 1 - sqrt( x_1 / g(x) ) ]`
/// with
///  `g(x) = 1 + 9 * ( Sum_i=2^n x_i ) / (n - 1)`
/// The problem vector (`x`) has `n=30` variables bounded to [0, 1]. The optional solution is
/// expected to bet 0 for all `x_i`.
///
/// Make sure to compile this in release mode to speed up the calculation:
///
/// `cargo run --example nsga2_zdt1 --release`
fn main() -> Result<(), Box<dyn Error>> {
    // Add log
    env_logger::builder().filter_level(LevelFilter::Info).init();

    // Load the built-in problem.
    let number_of_individuals: usize = 30;
    let problem = ZTD1Problem::create(number_of_individuals)?;

    // Setup and run the NSGA2 algorithm
    let args = NSGA2Arg {
        number_of_individuals,
        stopping_condition: StoppingConditionType::MaxGeneration(MaxGeneration(1000)),
        // use default options for the SBX and PM operators
        crossover_operator_options: None,
        mutation_operator_options: None,
        // no need to evaluate the objective in parallel
        parallel: Some(false),
        // do not export intermediate solutions
        resume_from_file: None,
        export_history: None,
        // to reproduce results
        seed: Some(10),
    };
    let mut algo = NSGA2::new(problem, args)?;
    algo.run()?;

    for (i, individual) in algo.get_results().individuals.iter().enumerate() {
        println!("Individual #{} {:?}", i + 1, individual.variables());
    }

    // Export serialised results at last generation
    let out_path = PathBuf::from(&env::current_dir().unwrap())
        .join("examples")
        .join("results");
    algo.save_to_json(&out_path, Some("ZDT1_2obj"))?;

    Ok(())
}
