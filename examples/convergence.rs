use std::env;
use std::path::PathBuf;

use optirustic::algorithms::{
    Algorithm, ExportHistory, MaxGeneration, NSGA2Arg, StoppingConditionType, NSGA2,
};
use optirustic::core::builtin_problems::SCHProblem;
use optirustic::core::OError;
use optirustic::metrics::HyperVolume;

/// This example shows how to track the algorithm convergence by calculating the hyper-volume metric.
/// This library employs different and fast methods, depending on the number of problem objectives,
/// such as [Fonseca et al. (2006)](http://dx.doi.org/10.1109/CEC.2006.1688440) or
/// [While et al. (2012)](http://dx.doi.org/10.1109/TEVC.2010.2077298).
///
/// Convergence can be tracked while an algorithm runs by saving the evolution history of the
/// population. This is useful if you have an evaluation  function, that evaluates the problem
/// objectives and constraints (this is the `evaluate()` function in the `Evaluator` trait),
/// that takes very long to run .
///
/// The example below solves the SCH problem, export the evolution history as JSON every 100
/// generations and calculate the hyper-volume at different generations.
///
/// Make sure to compile this in release mode to speed up the calculation:
///
/// `cargo run --example convergence --release`
fn main() -> Result<(), OError> {
    let problem = SCHProblem::create()?;
    let out_path = PathBuf::from(&env::current_dir().expect("Cannot fetch current directory"))
        .join("examples")
        .join("results")
        .join("convergence");

    // this exports a JSON file every 100 generations
    let export_history = ExportHistory::new(100, &out_path)?;
    let args = NSGA2Arg {
        number_of_individuals: 10,
        stopping_condition: StoppingConditionType::MaxGeneration(MaxGeneration(1000)),
        crossover_operator_options: None,
        mutation_operator_options: None,
        parallel: Some(false),
        export_history: Some(export_history),
        resume_from_file: None,
        seed: Some(10),
    };
    let mut algo = NSGA2::new(problem, args)?;
    algo.run()?;
    let mut results = algo.get_results();

    // select a reference point to use for the calculation. This is an arbitrary point that must be
    // dominated by all the points on the Pareto front. You can pick your own or estimate it with
    // the helper function below.
    let manual_pick = true;
    let ref_point = if !manual_pick {
        let offset = vec![10.0, 10.0]; // add an offset for the 2 objectives
        HyperVolume::estimate_reference_point(&results.individuals, Some(offset))?
    } else {
        vec![1000000.0, 1000000.0]
    };

    // calculate metric from the history file
    let serialised_data = NSGA2::read_json_file(&out_path.join("History_NSGA2_gen200.json"))?;
    let hv = HyperVolume::from_file(&serialised_data, &ref_point)?;
    println!(
        "Hyper-volume at generation #{} is {}",
        hv.generation, hv.value
    );

    // calculate metric at the last iteration
    let hv = HyperVolume::from_individual(&mut results.individuals, &ref_point)?;
    println!("Hyper-volume at last generation is {}", hv);

    // calculate the hyper-volume at all generations to check the overall convergence
    let all_serialise_data = NSGA2::read_json_files(&out_path)?;
    let hvs = HyperVolume::from_files(&all_serialise_data, &ref_point)?;
    println!("Hyper-volumes generations: {:?}", hvs.generations());
    println!("Hyper-volumes values: {:?}", hvs.values());

    Ok(())
}
