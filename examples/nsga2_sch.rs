use std::env;
use std::error::Error;
use std::path::PathBuf;

use log::LevelFilter;

use optirustic::algorithms::{
    Algorithm, MaxGenerationValue, NSGA2Arg, StoppingConditionType, NSGA2,
};
use optirustic::core::builtin_problems::SCHProblem;

/// Solve the Schaffer’s problem (SCH) where the following 2 objectives are minimised:
/// - `f_1(x) = x^2`
/// - `f-2(x) = (x - 2)^2`
///
/// The problem has 1 variable (`x`) bounded to -1000 and 1000. The optional solution is expected
/// to lie in the [0; 2] range. The algorithm converges in about 1 second.
///
/// Make sure to compile this in release mode to speed up the calculation:
///
/// `cargo run --example nsga2 -p optirustic --release`
fn main() -> Result<(), Box<dyn Error>> {
    // Add log
    env_logger::builder().filter_level(LevelFilter::Info).init();

    // Load the built-in problem.
    let problem = SCHProblem::create()?;

    // Setup and run the NSGA2 algorithm
    let args = NSGA2Arg {
        // use 100 individuals and stop the algorithm at 250 generations
        number_of_individuals: 100,
        stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(250)),
        // use default options for the SBX and PM operators
        crossover_operator_options: None,
        mutation_operator_options: None,
        // no need to evaluate the objective in parallel
        parallel: Some(false),
        // do not export intermediate solutions
        export_history: None,
        // to reproduce results
        resume_from_file: None,
        seed: Some(10),
    };
    let mut algo = NSGA2::new(problem, args)?;
    algo.run()?;

    let out_path = PathBuf::from(&env::current_dir().unwrap())
        .join("examples")
        .join("results");

    // Export serialised results at last generation
    algo.save_to_json(&out_path, Some("SCH_2obj"))?;

    // You can plot the data by running nsga2_sch_plot.py

    Ok(())
}
