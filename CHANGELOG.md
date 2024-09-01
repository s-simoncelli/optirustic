# Changelog

## 0.7.0

- Added new Python API to generate reference points with `DasDarren1998`. The new class
  allows getting the weights for the `NSGA3` algorithm and plotting them. See the Python
  type hints for name and description of the new class methods.
- Added `AdaptiveNSGA3` to use the adaptive approach to handle the reference points. This
  implements the new algorithm from Jain and Deb (2014) (doi.org/10.1109/TEVC.2013.2281534)
  to handle problems where not all reference points intersect the optimal Pareto front. This
  helps to reduce crowding and enhance the solution quality.

## 0.6.0

- Removed crate `hv-wfg-sys`. The hyper-volume from `HyperVolumeWhile2012` is now calculated
  using the Rust implementation of the While et al. (2012) approach from the paper. No public
  API has been changed.
- Replaced `GPL` license with `MIT` license.

## 0.5.0

- Remove `plot` feature and `plotters` as dependency. Charts can now be generated via
  the Python package `optirustic`
- Added python API for NSGA3 to plot reference points from the algorithm's data
- Added python function `plot_reference_points` to plot reference points from a vector
- Updated reference point examples to use new serialise function
- Added Python scripts to plot serialised reference points
- Added new charts in example folder generated from serialised data

## 0.4.1

- Added `Hypervolume::estimate_reference_point_from_file` and
  `Hypervolume::estimate_reference_point_from_files` methods, to easily
  estimate the reference point from a file or set of files.
- Python package API now includes the following new methods: `convergence_data`
  (to fetch the convergence data instead of plotting it), `estimate_reference_point_from_file`
  and `estimate_reference_point_from_files`.

## 0.4.0

- Made `Elapsed` fields public.
- Added `Constraint::target()` and `Constraint::operator()` to access the
  struct target and operator data respectively.
- Added `Variable::label` to get a label describing the type of variable set
  on a problem.
- Added `Individual::constraints()` and `Individual::objectives()` to access
  an individual's constraint and objective values grouped by their name.
- Added `Individual::data()` to access any custom data set on an individual
  (such as the rank, crowding distance or reference point distance).
- Added new `optirustic-py` crate to load serialised data from Python and
  plot Pareto front and convergence charts.

## 0.3.4

- Added `AlgorithmSerialisedExport::problem()` and `AlgorithmSerialisedExport::individuals()` to
  get the `Problem` and vector of `Individual` from serialised data.
- Implemented trait to convert from `AlgorithmSerialisedExport` to `AlgorithmExport`.

## 0.3.3

- Added `exported_on` field in `AlgorithmSerialisedExport`. This field
  contains the string with the date and time when the serialised data was exported
- Added `OError::File` to handle error messages related to files (for example when
  data is serialised or deserialised).
- Implemented `TryInto` trait to convert `ProblemExport` (from serialised data)
  to `Problem`.
- When serialised data are imported in `Algorithm::read_json_file`, the sign of
  objective values that are maximised is inverted. Maximised objectives are stored as
  negative values in the `Objective` struct and the sign is inverted back for the
  user in `Algorithm::save_to_json` when data is serialised.
- Fixed history export at desired generation step. The file was exported after 1
  or 2 generations instead.
- Added `Hypervolume` struct to calculate metric from serialised data and examples.
  The hyper-volume can now be calculated from the following sources:
    - an array of `Individual` using `HyperVolume::from_individual`
    - an array of objectives given as `f64` using `HyperVolume::from_values`
    - a JSON file using `HyperVolume::from_file`
    - a folder with JSON files using `HyperVolume::from_files`
- The `NSGA2` algorithm now uses `f64::MAX` instead of `f64::INFINITE` to handle
  points without a crowding distance. Infinite is not supported by serde and was
  converted to `null`.