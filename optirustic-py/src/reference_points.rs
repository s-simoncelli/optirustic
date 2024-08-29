use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use optirustic::utils::{DasDarren1998, NumberOfPartitions, TwoLayerPartitions};

use crate::get_plot_fun;

#[derive(Clone, Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub struct PyTwoLayerPartitions {
    pub boundary_layer: usize,
    pub inner_layer: usize,
    pub scaling: Option<f64>,
}

#[derive(Clone, Debug, FromPyObject)]
pub enum PyNumberOfPartitions {
    #[pyo3(transparent, annotation = "int")]
    OneLayer(usize),
    #[pyo3(transparent, annotation = "dict")]
    TwoLayers(PyTwoLayerPartitions),
}

#[pyclass(name = "DasDarren1998")]
#[derive(Clone)]
pub struct PyDasDarren1998 {
    number_of_objectives: usize,
    number_of_partitions: PyNumberOfPartitions,
}

// TODO number_of_partitions number or dict
#[pymethods]
impl PyDasDarren1998 {
    #[new]
    /// Initialise the class
    pub fn new(
        number_of_objectives: usize,
        number_of_partitions: PyNumberOfPartitions,
    ) -> PyResult<Self> {
        Ok(Self {
            number_of_objectives,
            number_of_partitions,
        })
    }

    pub fn calculate(&self) -> PyResult<Vec<Vec<f64>>> {
        let number_of_partitions = match &self.number_of_partitions {
            PyNumberOfPartitions::OneLayer(n) => NumberOfPartitions::OneLayer(*n),
            PyNumberOfPartitions::TwoLayers(data) => {
                NumberOfPartitions::TwoLayers(TwoLayerPartitions {
                    boundary_layer: data.boundary_layer,
                    inner_layer: data.inner_layer,
                    scaling: data.scaling,
                })
            }
        };
        let ds = DasDarren1998::new(self.number_of_objectives, &number_of_partitions)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(ds.get_weights())
    }

    /// Reference point plot from vector
    pub fn plot(&self, ref_points: Vec<Vec<f64>>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let fun: Py<PyAny> = get_plot_fun("plot_reference_points", py)?;
            fun.call1(py, (ref_points,))
        })
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DasDarren1998(number_of_objectives={}, number_of_partitions={:?})",
            self.number_of_objectives, self.number_of_partitions
        ))
    }

    pub fn __str__(&self) -> String {
        self.__repr__().unwrap()
    }
}
