use std::error::Error;
use std::path::PathBuf;

use log::LevelFilter;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::*;
use plotters::prelude::full_palette::{GREY_A400, RED_700};

use optirustic::algorithms::{Algorithm, MaxGeneration, NSGA2, NSGA2Arg, StoppingConditionType};
use optirustic::core::builtin_problems::SCHProblem;
use optirustic::core::Individual;

/// Solve the Schaffer’s problem (SCH) where the following 2 objectives are minimised:
/// - `f_1(x) = x^2`
/// - `f-2(x) = (x - 2)^2`
/// The problem has 1 variable (`x`) bounded to -0.001 and 0.001. The optional solution is expected
/// to lie in the [0; 2] range. The algorithm converges in about 1 second.
///
/// Make sure to compile this in release mode to speed up the calculation:
///
/// `cargo run --example nsga2 --release`
fn main() -> Result<(), Box<dyn Error>> {
    // Add log
    env_logger::builder().filter_level(LevelFilter::Info).init();

    // Load the built-in problem.
    let problem = SCHProblem::create()?;

    // Setup and run the NSGA2 algorithm
    let args = NSGA2Arg {
        // use 100 individuals and stop the algorithm at 250 generations
        number_of_individuals: 100,
        stopping_condition: StoppingConditionType::MaxGeneration(MaxGeneration(250)),
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

    // Plot the results
    plot(&algo.get_results().individuals)?;

    // Export serialised results at last generation
    algo.save_to_json(
        &PathBuf::from("optirustic/examples/results"),
        Some("SCH_2obj"),
    )?;

    Ok(())
}

/// Draw the expected objective functions and the solutions from the algorithm.
fn plot(individuals: &[Individual]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(
        "optirustic/examples/results/SCH_2_obj_NSGA2_solutions.png",
        (1024, 768),
    )
    .into_drawing_area();
    static FONT: &str = "sans-serif";

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(65)
        .y_label_area_size(65)
        .margin_top(10)
        .margin_left(10)
        .margin_right(30)
        .margin_bottom(5)
        .caption("SCH problem solved with NSGA2", (FONT, 30.0))
        .build_cartesian_2d(-5.0..5.0, -1.0..20.0)?;

    chart
        .configure_mesh()
        .y_desc("Objective")
        .x_desc("x")
        .axis_desc_style((FONT, 25, &BLACK))
        .label_style((FONT, 20, &BLACK))
        .draw()?;

    // Draw the expected objective functions
    let x: Vec<f64> = (-30000..30000).map(|i| i as f64 / 1000.0).collect();
    let f1: Vec<(f64, f64)> = x.iter().map(|xx| (*xx, SCHProblem::f1(*xx))).collect();
    let f2: Vec<(f64, f64)> = x.iter().map(|xx| (*xx, SCHProblem::f2(*xx))).collect();

    let color = Palette99::pick(10).to_rgba();
    chart
        .draw_series(LineSeries::new(
            f1,
            ShapeStyle {
                color,
                filled: false,
                stroke_width: 2,
            },
        ))?
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y + 1)], color))
        .label("Objective f1");

    let color = Palette99::pick(5).to_rgba();
    chart
        .draw_series(LineSeries::new(
            f2,
            ShapeStyle {
                color,
                filled: false,
                stroke_width: 2,
            },
        ))?
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y + 1)], color))
        .label("Objective f2");

    // Plot the solutions
    let points: Vec<(f64, f64, f64)> = individuals
        .iter()
        .map(|i| {
            let vars = i.get_variable_values().unwrap();
            let vars: Vec<f64> = vars.iter().map(|v| v.as_real().unwrap()).collect();
            let obj = i.get_objective_values().unwrap();
            (vars[0], obj[0], obj[1])
        })
        .collect();

    let dot_style = ShapeStyle {
        color: RED_700.to_rgba(),
        filled: true,
        stroke_width: 1,
    };
    chart
        .draw_series(
            points
                .iter()
                .map(|(x, o1, _)| Circle::new((*x, *o1), 5, dot_style)),
        )?
        .legend(move |(x, y)| {
            Rectangle::new([(x + 6, y - 6), (x + 16, y + 4)], dot_style.color.filled())
        })
        .label("Solution");
    chart.draw_series(
        points
            .iter()
            .map(|(x, _, o2)| Circle::new((*x, *o2), 5, dot_style)),
    )?;

    // Add legend box
    chart
        .configure_series_labels()
        .border_style(GREY_A400)
        .background_style(WHITE)
        .label_font((FONT, 20))
        .position(SeriesLabelPosition::Coordinate(2, 2))
        .draw()?;

    root.present()?;
    Ok(())
}
