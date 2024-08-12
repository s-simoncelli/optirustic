# Changelog

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