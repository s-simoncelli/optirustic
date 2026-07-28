use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use nsga_rs::algorithms::{
    Algorithm, ExportHistory, NSGA2Arg, NumThreads, StoppingCondition, NSGA2,
};
use nsga_rs::core::builtin_problems::SCHProblem;
use nsga_rs::core::OError;
use nsga_rs::metrics::HyperVolume;

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
/// `cargo run --example convergence -p nsga_rs --release`
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
        stopping_condition: StoppingCondition::MaxGeneration(1000),
        crossover_operator_options: None,
        mutation_operator_options: None,
        threads: NumThreads::Off,
        export_history: Some(export_history),
        resume_from_file: None,
        seed: Some(10),
    };
    let mut algo = NSGA2::new(Arc::new(problem), args)?;
    algo.run()?;
    let mut results = algo.get_results();

    let file_gen_200 = out_path.join("History_NSGA2_gen200.json");
    let serialised_data = NSGA2::read_json_file(&file_gen_200)?;
    let offset = Some(vec![100.0, 100.0]);
    let ref_point =
        HyperVolume::estimate_reference_point_from_file(&serialised_data, offset.clone())?;

    // calculate metric from the history file
    let serialised_data = NSGA2::read_json_file(&file_gen_200)?;
    let hv = HyperVolume::from_file(&serialised_data, &ref_point)?;
    println!(
        "Hyper-volume at generation #{} is {}",
        hv.generation, hv.value
    );

    // calculate ref point using all results
    let serialised_data = NSGA2::read_json_files(&out_path)?;
    let ref_point =
        HyperVolume::estimate_reference_point_from_files(&serialised_data, offset.clone())?;
    println!("Reference point: {:?}", ref_point);

    // calculate metric at the last iteration
    let hv = HyperVolume::from_individuals(&mut results.individuals, &ref_point)?;
    println!("Hyper-volume at last generation is {}", hv);

    // calculate the hyper-volume at all generations to check the overall convergence
    let all_serialise_data = NSGA2::read_json_files(&out_path)?;
    let hvs = HyperVolume::from_files(&all_serialise_data, &ref_point)?;
    println!("Hyper-volumes generations: {:?}", hvs.generations());
    println!("Hyper-volumes values: {:?}", hvs.values());

    // plot the convergence/hyper-volume chart
    HyperVolume::plot_from_files(
        &serialised_data,
        &ref_point,
        &out_path
            .parent()
            .unwrap()
            .join("SCH_2obj_NSGA2_convergence_chart.png"),
    )?;

    Ok(())
}
