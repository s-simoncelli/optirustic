use std::cmp::Ordering;

/// A struct defining the objective values for an individual.
#[derive(Clone, Debug)]
struct Individual(Vec<f64>);

impl Individual {
    /// Get an objective value for the individual.
    ///
    /// # Arguments
    ///
    /// * `index`: The objective index.
    ///
    /// returns: `Result<f64, String>`: The objective or a string with the error if the `index` does
    /// not exist.
    fn obj(&self, index: usize) -> Result<f64, String> {
        match self.0.get(index) {
            None => Err(format!("The objective index {} does not exist", index)),
            Some(v) => Ok(*v),
        }
    }
}

#[derive(Clone, Debug)]
/// A front with individuals.
struct Front {
    /// The vector with the individuals and their objectives.
    points: Vec<Individual>,
    /// The number of individuals in the front.
    number_of_individuals: usize,
    /// The optimisation level
    opt: Optimisation,
}

impl Front {
    /// Creates a front with objectives with value 0.0.
    ///
    /// # Arguments
    ///
    /// * `number_of_individuals`: The number of individuals to add.
    /// * `number_of_objectives`: The number of objectives to add for each individual.
    /// * `opt`: The optimisation level to use when sorting the data.
    ///
    /// returns: `Front`
    fn empty(number_of_individuals: usize, number_of_objectives: usize, opt: Optimisation) -> Self {
        let empty_ind = Individual(vec![0.0; number_of_objectives]);
        let points = vec![empty_ind; number_of_individuals];
        Front {
            points,
            number_of_individuals,
            opt,
        }
    }

    /// Get the `Individual` containing its objective values.
    ///
    /// # Arguments
    ///
    /// * `index`: The individual index.
    ///
    /// returns: `Result<&Individual, String>`: The `Individual` or a string with the error if
    /// the `index` does not exist.
    fn ind(&self, index: usize) -> Result<&Individual, String> {
        match self.points.get(index) {
            None => Err(format!("The individual index {} does not exist", index)),
            Some(v) => Ok(v),
        }
    }

    /// Sort the front objectives.
    ///
    /// # Arguments
    ///
    /// * `depth`: The depth of the optimisation (only used when `opt` is `1`).
    /// * `obj_count`: The number of objectives to process.
    ///
    /// returns: `Result<(), String> `
    fn sort(&mut self, depth: i8, obj_count: usize) -> Result<(), String> {
        let mut did_swap = true;
        let mut temp;
        while did_swap {
            did_swap = false;
            if self.number_of_individuals < 2 {
                return Ok(());
            }
            for i in 0..=self.number_of_individuals - 2 {
                if self.greater(&self.points[i], &self.points[i + 1], depth, obj_count)? {
                    did_swap = true;
                    temp = self.points[i].clone();
                    self.points[i] = self.points[i + 1].clone();
                    self.points[i + 1] = temp;
                }
            }
        }
        Ok(())
    }

    fn update(&mut self, ind: usize, obj: usize, value: f64) -> Result<(), String> {
        let ind = self
            .points
            .get_mut(ind)
            .ok_or(format!("Cannot find individual index {}", ind))?;
        let obj = ind
            .0
            .get_mut(obj)
            .ok_or(format!("Cannot find objective index {}", obj))?;
        *obj = value;

        Ok(())
    }

    /// Sorting function by dominance.
    ///
    /// # Arguments
    ///
    /// * `v1`: The first individual.
    /// * `v2`: The second individual.
    /// * `depth`: The depth of the optimisation (only used when `opt` is `1`).
    /// * `obj_count`: The number of objectives to process.
    ///
    /// returns: `Result<bool, String>`
    fn greater(
        &self,
        v1: &Individual,
        v2: &Individual,
        depth: i8,
        obj_count: usize,
    ) -> Result<bool, String> {
        let it = if self.opt.value() == 1 {
            let max_it = obj_count - depth as usize - 1;
            (0..=max_it).rev()
        } else {
            (0..=obj_count - 1).rev()
        };

        for i in it {
            return if v1.obj(i)? < v2.obj(i)? {
                Ok(true)
            } else if v2.obj(i)? < v1.obj(i)? {
                Ok(false)
            } else {
                continue;
            };
        }
        Ok(false)
    }
}

/// A set of fronts used in the calculation of the hyper-volume.
#[derive(Debug)]
struct FrontSet {
    /// The vector with the front sets being processes.
    fronts: Vec<Front>,
    /// The current recursion depth of the set.
    depth: i8,
    /// The maximum recursion depth allocated so far (for `opt.value()` == `0`).
    max_depth: i8,
}

impl FrontSet {
    ///Get the front at the previous depth.
    ///
    /// returns: `&Front`
    fn prev(&self) -> &Front {
        &self.fronts[self.depth as usize - 1]
    }

    /// Increase the current depth counter by 1.
    fn increase_depth(&mut self) {
        self.depth += 1;
    }

    /// Decrease the current depth counter by 1.
    fn decrease_depth(&mut self) {
        self.depth -= 1;
    }
}

#[derive(PartialEq, Clone, Debug)]
/// The level of optimisation to use when calculating the hyper-volume.
pub enum Optimisation {
    /// Do not apply any optimisation
    O0,
    O1,
    O2,
    O3,
}

impl PartialOrd for Optimisation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value().partial_cmp(&other.value())
    }
}

impl Optimisation {
    /// Get the optimisation value.
    ///
    /// return: `u8`
    pub fn value(&self) -> u8 {
        match &self {
            Optimisation::O0 => 0,
            Optimisation::O1 => 1,
            Optimisation::O2 => 2,
            Optimisation::O3 => 3,
        }
    }
}

/// The dominance relationship between two individuals.
enum Dominance {
    First,
    Second,
    Equal,
    Skip,
}

///
/// This struct implements the high performance algorithm by While et al. (2010) for a set of
/// non-dominated points.
///
/// > Lyndon While, Lucas Bradstreet, and Luigi Barone. A Fast Way of Calculating Exact
/// > Hypervolumes. IEEE Transactions on Evolutionary Computation 16(1), 2012.
pub struct Wfg {
    /// The reference point
    reference_point: Vec<f64>,
    /// The number of points or individuals in the front.
    number_of_individuals: usize,
    /// The number of objectives.
    number_of_objectives: usize,
    /// The front data whose hyper-volume is calculated.
    f: Front,
    /// The optimisation level.
    opt: Optimisation,
}

impl Wfg {
    /// Initialise the `Wfg` structure to calculate hyper-volume.
    ///
    /// # Arguments
    ///
    /// * `objective_values`: The objective values. Each item is an individual which contains
    ///    a vector with size equal to the number of problem objectives.
    /// * `ref_point`: The reference point.
    /// * `opt`: The optimisation level.
    ///
    /// returns: `Wfg`
    pub fn new(objective_values: &[Vec<f64>], ref_point: &[f64], opt: Optimisation) -> Self {
        let number_of_individuals = objective_values.len();
        let number_of_objectives = ref_point.len();

        let f: Front = Front {
            points: objective_values
                .iter()
                .map(|ind_values| Individual(ind_values.clone()))
                .collect(),
            number_of_individuals,
            opt: opt.clone(),
        };

        Self {
            reference_point: ref_point.to_vec(),
            number_of_individuals,
            number_of_objectives,
            f,
            opt,
        }
    }

    /// Calculate the hyper-volume with the method.
    ///
    /// returns: `Result<f64, String>`
    pub fn calculate(&self) -> Result<f64, String> {
        let mut fs: Vec<Front>;
        if self.opt == Optimisation::O0 {
            let empty_front = Front::empty(
                self.number_of_individuals,
                self.number_of_objectives,
                self.opt.clone(),
            );
            fs = vec![empty_front; self.number_of_individuals];
        } else {
            // enable slicing to save a level of recursion. max_d is 0 if opt != 3
            let max_obj: usize =
                self.number_of_objectives - (self.opt.value() as f32 / 2.0 + 1.0).floor() as usize;
            fs = Vec::<Front>::with_capacity(max_obj);

            // 3D base needs space for the sentinels. max_p is zero when opt < 3
            let max_ind =
                self.number_of_individuals + 2 * (self.opt.value() as f32 / 3.0).floor() as usize;

            for i in 0..max_obj {
                let new_obj_num = self.number_of_objectives
                    - (i + 1) * (self.opt.value() as f32 / 2.0).floor() as usize;
                let empty_front = Front::empty(max_ind, new_obj_num, self.opt.clone());
                fs.push(empty_front);
            }
        }

        let mut front_sets = FrontSet {
            fronts: fs,
            depth: 0,
            max_depth: -1,
        };

        let mut f = self.f.clone();
        if self.opt == Optimisation::O3 {
            return if self.number_of_objectives == 2 {
                f.sort(0, self.number_of_objectives)?;
                self.volume_2d(&f)
            } else {
                self.volume(&mut front_sets, &mut f, self.number_of_objectives)
            };
        }

        self.volume(&mut front_sets, &mut f, self.number_of_objectives)
    }

    /// Return the hyper-volume for a 2D front.
    ///
    /// # Arguments
    ///
    /// * `front`: The front with the objective values.
    ///
    /// returns: `f64`
    fn volume_2d(&self, front: &Front) -> Result<f64, String> {
        let x0 = front.ind(0)?.obj(0)?;
        let y0 = front.ind(0)?.obj(1)?;
        let mut volume = ((x0 - self.reference_point[0]) * (y0 - self.reference_point[1])).abs();

        for i in 1..front.number_of_individuals {
            volume += ((front.ind(i)?.obj(0)? - self.reference_point[0])
                * (front.ind(i)?.obj(1)? - front.ind(i - 1)?.obj(1)?))
            .abs();
        }

        Ok(volume)
    }

    /// Calculate the hyper-volume for a `front`.
    ///
    /// # Arguments
    ///
    /// * `set`: The front set.
    /// * `front`: The front being processed.
    /// * `obj_count`: The number of objectives.
    ///
    /// returns: `Result<f64, String>`
    fn volume(
        &self,
        set: &mut FrontSet,
        front: &mut Front,
        obj_count: usize,
    ) -> Result<f64, String> {
        if self.opt > Optimisation::O0 {
            front.sort(set.depth, obj_count)?;
        }

        if (self.opt == Optimisation::O2) & (obj_count == 2) {
            return self.volume_2d(front);
        }

        let mut volume = 0.0;
        if self.opt <= Optimisation::O1 {
            for i in 0..front.number_of_individuals {
                volume += self.exclusive_hv(set, front, i, obj_count)?;
            }
        } else {
            let obj_count = obj_count - 1;
            for i in (0..front.number_of_individuals).rev() {
                volume += (front.ind(i)?.obj(obj_count)? - self.reference_point[obj_count]).abs()
                    * self.exclusive_hv(set, front, i, obj_count)?;
            }
        }

        Ok(volume)
    }

    /// Calculate the exclusive hyper-volume of a point indexed by `p` in the
    /// front `front` relative to the set `fs`.
    ///
    /// # Arguments
    ///
    /// * `set`: The front set.
    /// * `front`: The front being processed.
    /// * `ind_idx`: The individual index being processed.
    /// * `obj_count`: The number of objectives.
    ///
    /// returns: `Result<f64, String>`
    fn exclusive_hv(
        &self,
        set: &mut FrontSet,
        front: &Front,
        ind_idx: usize,
        obj_count: usize,
    ) -> Result<f64, String> {
        let mut volume = self.inclusive_hv(front.ind(ind_idx)?, obj_count)?;

        if front.number_of_individuals > ind_idx + 1 {
            self.limit_set(set, front, ind_idx, obj_count)?;
            let mut prev_front = set.prev().clone();
            volume -= self.volume(set, &mut prev_front, obj_count)?;
            set.decrease_depth();
        }
        Ok(volume)
    }

    /// Calculate the inclusive hyper-volume for an individual.
    ///
    /// # Arguments
    ///
    /// * `individual`: The individual
    /// * `obj_count`: The number of objectives.
    ///
    /// returns: `Result<f64, String>`
    fn inclusive_hv(&self, individual: &Individual, obj_count: usize) -> Result<f64, String> {
        let mut volume = 1.0;
        for i in 0..obj_count {
            volume *= (individual.obj(i)? - self.reference_point[i]).abs();
        }
        Ok(volume)
    }

    /// Build the non-dominated sub-set.
    ///
    /// # Arguments
    ///
    /// * `set`: The front set.
    /// * `front`: The front being processed.
    /// * `p`: The individual's index being processed.
    /// * `obj_count`: The number of objectives.
    ///
    /// returns: `Result<(), String>`
    fn limit_set(
        &self,
        set: &mut FrontSet,
        front: &Front,
        p: usize,
        obj_count: usize,
    ) -> Result<(), String> {
        let d = set.depth as usize;
        if self.opt == Optimisation::O0 && ((set.depth > set.max_depth) || set.depth == 0) {
            set.max_depth = set.depth;
            set.fronts[d] = Front::empty(self.number_of_individuals, obj_count, self.opt.clone());
        }

        let max_rec_depth = front.number_of_individuals - 1 - p;
        for ind_idx in 0..max_rec_depth {
            for obj_idx in 0..obj_count {
                set.fronts[d].update(
                    ind_idx,
                    obj_idx,
                    self.get_dominated(
                        front.ind(p)?.obj(obj_idx)?,
                        front.ind(p + 1 + ind_idx)?.obj(obj_idx)?,
                    ),
                )?;
            }
        }
        set.fronts[d].number_of_individuals = 1;

        for ind_idx in 1..max_rec_depth {
            let mut obj_idx: usize = 0;
            let mut preserve_ind = true;
            while (obj_idx < set.fronts[d].number_of_individuals) && preserve_ind {
                match self.nds(
                    set.fronts[d].ind(ind_idx)?,
                    set.fronts[d].ind(obj_idx)?,
                    d,
                    obj_count,
                )? {
                    Dominance::First => {
                        set.fronts[d].number_of_individuals -= 1;
                        let t = set.fronts[d].ind(obj_idx)?.clone();
                        let ind_count = set.fronts[d].number_of_individuals;
                        set.fronts[d].points[obj_idx] = set.fronts[d].ind(ind_count)?.clone();
                        set.fronts[d].points[ind_count] = t;
                    }
                    Dominance::Skip => {
                        obj_idx += 1;
                    }
                    _ => {
                        preserve_ind = false;
                    }
                }
            }

            // add the individual to the set
            if preserve_ind {
                let ind_count = set.fronts[d].number_of_individuals;
                let t = set.fronts[d].ind(ind_count)?.clone();
                set.fronts[d].points[ind_count] = set.fronts[d].ind(ind_idx)?.clone();
                set.fronts[d].points[ind_idx] = t;
                set.fronts[d].number_of_individuals += 1;
            }
        }

        // increase the depth to process another set
        set.increase_depth();
        Ok(())
    }

    /// Get the non-dominated individual.
    ///
    /// # Arguments
    ///
    /// * `i1`: The first individual.
    /// * `q`: The second individual.
    /// * `depth`: The recursion depth.
    /// * `obj_count`: The number of objectives.
    ///
    /// returns: `Result<i8, String>`
    fn nds(
        &self,
        i1: &Individual,
        i2: &Individual,
        depth: usize,
        obj_count: usize,
    ) -> Result<Dominance, String> {
        let it = if self.opt == Optimisation::O1 {
            (0..obj_count - depth).rev()
        } else {
            (0..obj_count).rev()
        };
        for i in it {
            if Self::dominates(i1.obj(i)?, i2.obj(i)?) {
                for j in (0..i).rev() {
                    if Self::dominates(i2.obj(j)?, i1.obj(j)?) {
                        return Ok(Dominance::Skip);
                    }
                }
                return Ok(Dominance::First);
            } else if Self::dominates(i2.obj(i)?, i1.obj(i)?) {
                for j in (0..i).rev() {
                    if Self::dominates(i1.obj(j)?, i2.obj(j)?) {
                        return Ok(Dominance::Skip);
                    }
                }
                return Ok(Dominance::Second);
            }
        }
        Ok(Dominance::Equal)
    }

    /// Get the dominated value between of two objective values from two individuals.
    ///
    /// # Arguments
    ///
    /// * `o1`: The first objective value.
    /// * `o2`: The second objective value.
    ///
    /// returns: `f64`
    /// ```
    fn get_dominated(&self, o1: f64, o2: f64) -> f64 {
        if Self::dominates(o2, o1) {
            o1
        } else {
            o2
        }
    }

    /// Check whether `o1` dominates `o2`.
    ///
    /// # Arguments
    ///
    /// * `o1`: The first objective value.
    /// * `o2`: The second objective value.
    ///
    /// returns: `bool`
    fn dominates(o1: f64, o2: f64) -> bool {
        o1 < o2
    }
}

#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;

    use crate::metrics::hv_wfg::wfg::{Optimisation, Wfg};

    #[test]
    fn test() {
        let ref_point = vec![10.0; 3];
        let data = vec![
            vec![0.500999867734, 0.501000000033, 0.500999987997],
            vec![9.84167759049e-09, 2.36154644108e-09, 0.499999987997],
            vec![0.499999867734, 1.32416636196e-07, 3.33066907488e-16],
            vec![2.52520317534e-18, 2.01754168497e-08, 0.499999979974],
            vec![3.06183729901e-12, 0.500000000033, 0.0],
        ];
        let hv = Wfg::new(&data, &ref_point, Optimisation::O2);

        assert_approx_eq!(f64, hv.calculate().unwrap(), 999.874999, epsilon = 0.0001);
    }
}
