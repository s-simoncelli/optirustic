use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use gnuplot::PlotOption::{Caption, Color, LineWidth, PointSymbol};
use gnuplot::{AutoOption, AxesCommon, Figure};
use log::LevelFilter;

use nsga_rs::algorithms::{
    Algorithm, AlgorithmExport, NSGA2Arg, NumThreads, StoppingCondition, NSGA2,
};
use nsga_rs::core::builtin_problems::SCHProblem;

/// Solve the Schaffer’s problem (SCH) where the following 2 objectives are minimised:
/// - `f_1(x) = x^2`
/// - `f_2(x) = (x - 2)^2`
///
/// The problem has 1 variable (`x`) bounded to -1000 and 1000. The optional solution is expected
/// to lie in the [0; 2] range. The algorithm converges in about 1 second.
///
/// Make sure to compile this in release mode to speed up the calculation:
///
/// `cargo run --example nsga2 -p nsga_rs --feature plotting --release`
fn main() -> Result<(), Box<dyn Error>> {
    // Add log
    env_logger::builder().filter_level(LevelFilter::Info).init();

    // Load the built-in problem.
    let problem = SCHProblem::create()?;

    // Setup and run the NSGA2 algorithm
    let args = NSGA2Arg {
        // use 100 individuals and stop the algorithm at 250 generations
        number_of_individuals: 100,
        stopping_condition: StoppingCondition::MaxGeneration(250),
        // use default options for the SBX and PM operators
        crossover_operator_options: None,
        mutation_operator_options: None,
        // no need to evaluate the objective in parallel
        threads: NumThreads::Off,
        // do not export intermediate solutions
        export_history: None,
        // to reproduce results
        resume_from_file: None,
        seed: Some(10),
    };
    let mut algo = NSGA2::new(Arc::new(problem), args)?;
    algo.run()?;

    // Export serialised results at last generation
    let out_path = PathBuf::from(&env::current_dir().unwrap())
        .join("examples")
        .join("results");
    algo.save_to_json(&out_path, Some("SCH_2obj"))?;

    // Plot the Pareto front using the exported data at the last generation
    let data_file = out_path.join("SCH_2obj_NSGA2_gen250.json");
    let serialised_data = NSGA2::read_json_file(&data_file)?;
    serialised_data.plot_front(&out_path.join("SCH_2obj_NSGA2_Pareto_front.png"), None)?;

    // Plot the solution chart SCH_2obj_NSGA2_solutions.png
    let export_data: AlgorithmExport = serialised_data.try_into()?;

    let x = (-60..=60).map(|v| v as f64 / 10.0).collect::<Vec<_>>();
    let f1: Vec<f64> = x.iter().map(|v| v.powi(2)).collect();
    let f2: Vec<f64> = x.iter().map(|v| (v - 2.0).powi(2)).collect();

    let mut fg = Figure::new();
    let ax = fg.axes2d();

    // theoretical curves
    ax.lines(
        &x,
        &f1,
        &[
            Color("blue".into()),
            Caption("Objective f_1"),
            LineWidth(2.0),
        ],
    );
    ax.lines(
        &x,
        f2,
        &[
            Color("green".into()),
            Caption("Objective f_2"),
            LineWidth(2.0),
        ],
    );

    // data from algorithm
    let var = export_data.get_real_variables("x")?;
    let calc_f1 = export_data.get_objective("x^2")?;
    let calc_f2 = export_data.get_objective("(x-2)^2")?;
    ax.points(
        &var,
        calc_f1,
        &[Color("red".into()), PointSymbol('O'), Caption("Solution")],
    );
    ax.points(&var, calc_f2, &[Color("red".into()), PointSymbol('O')]);

    ax.set_x_label("x", &[])
        .set_y_label("Objective", &[])
        .set_x_grid(true)
        .set_x_range(AutoOption::Fix(-3.0), AutoOption::Fix(5.0))
        .set_y_grid(true)
        .set_y_range(AutoOption::Fix(0.0), AutoOption::Fix(20.0))
        .set_title("SCH problem solved with NSGA2", &[]);
    fg.save_to_png(
        &data_file
            .parent()
            .unwrap()
            .join("SCH_2obj_NSGA2_solutions.png"),
        600,
        500,
    )?;
    Ok(())
}
