pub use algebra::{dot_product, perpendicular_distance, solve_linear_system, vector_magnitude};
pub use fast_non_dominated_sort::{fast_non_dominated_sort, NonDominatedSortResults};
pub use reference_points::DasDarren1998;
pub use vectors::{all_close, argmin, argsort, Sort, vector_max, vector_min};

mod algebra;
mod fast_non_dominated_sort;
mod reference_points;
mod vectors;
