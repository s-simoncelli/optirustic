use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use gnuplot::PlotOption::{Caption, Color, PointSize, PointSymbol};
use gnuplot::{AxesCommon, Figure};
use log::LevelFilter;

use nsga_rs::algorithms::{
    Algorithm, AlgorithmExport, ExportVecGroupBy, NSGA3Arg, Nsga3NumberOfIndividuals, NumThreads,
    StoppingCondition, NSGA3,
};
use nsga_rs::core::builtin_problems::DTLZ1Problem;
use nsga_rs::operators::SimulatedBinaryCrossoverArgs;
use nsga_rs::utils::{DasDarren1998, NumberOfPartitions};

/// Solve the DTLZ1 problem from Deb et al. (2013) with 3 objectives. This is a problem where the
/// optimal solutions or objectives lie on the hyper-plane passing through the intercept point
/// at 0.5 on each objective axis. This code replicates the first testing problem in Deb et al.
/// (2013).
///
/// Make sure to compile this in release mode to speed up the calculation:
///
/// `cargo run --example nsga3_dtlz1 -p nsga_rs --feature plotting --release`
fn main() -> Result<(), Box<dyn Error>> {
    // Add log
    env_logger::builder().filter_level(LevelFilter::Info).init();

    let number_objectives: usize = 3;
    // Set the number of variables to use in the DTLZ1 problem
    let k: usize = 5;
    let number_variables: usize = number_objectives + k - 1;
    // Get the built-in problem
    let problem = DTLZ1Problem::create(number_variables, number_objectives, false)?;

    // Set the number of partitions to create the reference points for the NSGA3 algorithm. This
    // uses one layer of 12 uniform gaps
    let number_of_partitions = NumberOfPartitions::OneLayer(12);
    // NSGA3 internally uses the Das & Darren approach to generate the points. This is also
    // available using:
    let das_darren = DasDarren1998::new(number_objectives, &number_of_partitions)?;
    println!(
        "Number of reference points to generate: {}",
        das_darren.number_of_points()
    );

    // Customise the SBX and PM operators like in the paper
    let crossover_operator_options = SimulatedBinaryCrossoverArgs {
        distribution_index: 30.0,
        crossover_probability: 1.0,
        ..SimulatedBinaryCrossoverArgs::default()
    };

    // Set up the NSGA3 algorithm
    let args = NSGA3Arg {
        // number of individuals from the paper (possibly equal to number of reference points)
        number_of_individuals: Nsga3NumberOfIndividuals::Custom(92),
        number_of_partitions,
        crossover_operator_options: Some(crossover_operator_options),
        mutation_operator_options: None,
        // stop at generation 400
        stopping_condition: StoppingCondition::MaxGeneration(400),
        threads: NumThreads::Off,
        export_history: None,
        // to reproduce results
        seed: Some(1),
        resume_from_file: None,
    };

    // Initialise the algorithm
    let mut algo = NSGA3::new(Arc::new(problem), args, false).unwrap();

    // Run the algorithm
    algo.run()?;

    // Export the last results to a JSON file
    let destination = PathBuf::from(&env::current_dir().unwrap())
        .join("examples")
        .join("results");

    algo.save_to_json(&destination, Some("DTLZ1_3obj"))?;

    // Plot the Pareto front using the exported data at the last generation
    let serialised_data = NSGA3::read_json_file(&destination.join("DTLZ1_3obj_NSGA3_gen400.json"))?;
    serialised_data.plot_front(
        &destination.join("DTLZ1_3obj_NSGA3_gen400_Pareto_front.png"),
        None,
    )?;

    // Plot the reference points against the normalised objective values
    let export_data: AlgorithmExport = serialised_data.try_into()?;
    let obj_names = export_data.problem.objective_names();

    // Generate a chart with the normalised objectives against the reference points
    // The plane is limited in the [0, 1] range.
    let mut fg = Figure::new();
    let ax = fg.axes3d();

    // Plot reference points
    let ref_points = export_data.get_nsga3_reference_points(ExportVecGroupBy::Objective)?;
    ax.points(
        &ref_points[0],
        &ref_points[1],
        &ref_points[2],
        &[
            PointSymbol('x'),
            PointSize(2.0),
            Color("red".into()),
            Caption("Reference points objectives"),
        ],
    );

    let all_normalised_objectives: Vec<_> =
        export_data.get_nsga3_normalised_objectives(ExportVecGroupBy::Objective)?;
    ax.points(
        &all_normalised_objectives[0],
        &all_normalised_objectives[1],
        &all_normalised_objectives[2],
        &[
            PointSymbol('O'),
            PointSize(0.5),
            Color("black".into()),
            Caption("Normalised objectives"),
        ],
    );

    ax.set_x_label(&obj_names[0], &[])
        .set_y_label(&obj_names[1], &[])
        .set_z_label(&obj_names[2], &[])
        .set_x_grid(true)
        .set_y_grid(true)
        .set_z_grid(true)
        .set_view(60.0, 110.0)
        .set_title(
            &format!(
                "Normalised objectives vs. reference points\nfor NSGA3 @ generation={}",
                export_data.generation
            ),
            &[],
        );

    fg.save_to_png(
        &destination.join("DTLZ1_3obj_NSGA3_gen400_obj_vs_ref_points.png"),
        800,
        600,
    )?;

    Ok(())
}
