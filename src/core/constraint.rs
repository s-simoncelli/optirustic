use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

/// Operator used to check a bounded constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationalOperator {
    /// Value must equal the constraint value
    EqualTo,
    /// Value must not equal the constraint value
    NotEqualTo,
    /// Value must be less or equal to the constraint value
    LessOrEqualTo,
    /// Value must be less than the constraint value
    LessThan,
    /// Value must be greater or equal to the constraint value
    GreaterOrEqualTo,
    /// Value must be greater than the constraint value
    GreaterThan,
}

/// Define a constraint where a value is compared with a relational operator as follows:
///  - Equality operator ([`RelationalOperator::EqualTo`]): value == target
///  - Inequality operator ([`RelationalOperator::NotEqualTo`]): value != target
///  - Greater than operator ([`RelationalOperator::GreaterThan`]): value > target
///  - Greater or equal to operator ([`RelationalOperator::GreaterOrEqualTo`]): value >= target
///  - less than operator ([`RelationalOperator::LessThan`]): value < target
///  - less or equal to operator ([`RelationalOperator::LessOrEqualTo`]): value <= target
///
/// # Example
///
/// ```
///   use optirustic::core::{Constraint, RelationalOperator};
///   let c = Constraint::new("Z>=5.2",RelationalOperator::GreaterOrEqualTo, 5.2);
///   assert_eq!(c.is_met(10.1), true);
///   assert_eq!(c.is_met(3.11), false);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constraint {
    /// The constraint name.
    name: String,
    /// The relational operator to use to compare a value against the constraint target value.
    operator: RelationalOperator,
    /// The constraint target.
    target: f64,
}

impl Constraint {
    /// Create a new relational constraint.
    ///
    /// # Arguments
    ///
    /// * `name`: The constraint name.
    /// * `operator`: The relational operator to use to compare a value against the constraint
    ///    target value.
    /// * `target`: The constraint target.
    ///
    /// returns: `Constraint`
    pub fn new(name: &str, operator: RelationalOperator, target: f64) -> Self {
        Self {
            name: name.to_owned(),
            operator,
            target,
        }
    }

    /// Create a new relational constraint with a scale and offset. The target is first scaled,
    /// then offset.
    ///
    /// # Arguments
    ///
    /// * `name`: The constraint name.
    /// * `operator`: The relational operator to use to compare a value against the constraint
    ///    target value.
    /// * `target`: The constraint target.
    /// * `scale`: Apply a scaling factor to the `target`.
    /// * `offset`: Apply an offset to the `target`.
    ///
    /// returns: `Constraint`
    pub fn new_with_modifiers(
        name: &str,
        operator: RelationalOperator,
        target: f64,
        scale: f64,
        offset: f64,
    ) -> Self {
        Self {
            name: name.to_owned(),
            operator,
            target: target * scale + offset,
        }
    }

    /// Get the constraint name.
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Check whether the constraint is met. This is assessed as follows:
    ///  - Equality operator ([`RelationalOperator::EqualTo`]): value == target
    ///  - Inequality operator ([`RelationalOperator::NotEqualTo`]): value != target
    ///  - Greater than operator ([`RelationalOperator::GreaterThan`]): value > target
    ///  - Greater or equal to operator ([`RelationalOperator::GreaterOrEqualTo`]): value >= target
    ///  - less than operator ([`RelationalOperator::LessThan`]): value < target
    ///  - less or equal to operator ([`RelationalOperator::LessOrEqualTo`]): value <= target
    ///
    /// # Arguments
    ///
    /// * `value`: The value to check against the constraint target.
    ///
    /// returns: `bool`
    pub fn is_met(&self, value: f64) -> bool {
        match self.operator {
            RelationalOperator::EqualTo => value == self.target,
            RelationalOperator::NotEqualTo => value != self.target,
            RelationalOperator::LessOrEqualTo => value <= self.target,
            RelationalOperator::LessThan => value < self.target,
            RelationalOperator::GreaterOrEqualTo => value >= self.target,
            RelationalOperator::GreaterThan => value > self.target,
        }
    }

    /// Calculate the amount of violation of the constraint for a solution value. This is a measure
    /// about how close (or far) the constraint value is from the constraint target. If the
    /// constraint is met (i.e. the solution associated to the constraint is feasible), then the
    /// violation is 0.0. Otherwise, the absolute difference between `target` and `value`
    /// is returned.
    ///
    /// See:
    ///  - Kalyanmoy Deb & Samir Agrawal. (2002). <https://doi.org/10.1007/978-3-7091-6384-9_40>.
    ///  - Shuang Li, Ke Li, Wei Li. (2022). <https://doi.org/10.48550/arXiv.2205.14349>.
    ///
    /// # Arguments
    ///
    /// * `value`: The value to check against the constraint target.
    ///
    /// return: `f64`
    pub fn constraint_violation(&self, value: f64) -> f64 {
        if self.is_met(value) {
            0.0
        } else {
            match self.operator {
                RelationalOperator::EqualTo => f64::abs(self.target - value),
                RelationalOperator::NotEqualTo => 1.0,
                RelationalOperator::LessOrEqualTo | RelationalOperator::GreaterOrEqualTo => {
                    f64::abs(self.target - value)
                }
                RelationalOperator::LessThan | RelationalOperator::GreaterThan => {
                    // add the tolerance
                    f64::abs(self.target - value) + 0.0001
                }
            }
        }
    }

    /// Get the set constraint target.
    ///
    /// returns: `f64`.
    pub fn target(&self) -> f64 {
        self.target
    }

    /// Get the set constraint operator.
    ///
    /// returns: `f64`.
    pub fn operator(&self) -> RelationalOperator {
        self.operator.clone()
    }
}

impl Display for Constraint {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let sign = match self.operator {
            RelationalOperator::EqualTo => "==",
            RelationalOperator::NotEqualTo => "!=",
            RelationalOperator::LessOrEqualTo => "<=",
            RelationalOperator::LessThan => "<",
            RelationalOperator::GreaterOrEqualTo => ">=",
            RelationalOperator::GreaterThan => ">",
        };
        f.write_fmt(format_args!("{} {} {}", self.name, sign, self.target))
    }
}

#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;

    use crate::core::{Constraint, RelationalOperator};

    #[test]
    fn test_is_met() {
        let c = Constraint::new("test", RelationalOperator::EqualTo, 5.2);
        assert!(c.is_met(5.2));
        assert!(!c.is_met(15.0));

        let c = Constraint::new("test", RelationalOperator::NotEqualTo, 5.2);
        assert!(!c.is_met(5.2));
        assert!(c.is_met(15.0));

        let c = Constraint::new("test", RelationalOperator::GreaterThan, 5.2);
        assert!(!c.is_met(5.2));
        assert!(c.is_met(15.0));
        assert!(!c.is_met(1.0));

        let c = Constraint::new("test", RelationalOperator::GreaterOrEqualTo, 5.2);
        assert!(c.is_met(5.2));
        assert!(c.is_met(15.0));
        assert!(!c.is_met(1.0));

        let c = Constraint::new("test", RelationalOperator::LessThan, 5.2);
        assert!(!c.is_met(5.2));
        assert!(c.is_met(1.0));
        assert!(!c.is_met(15.0));

        let c = Constraint::new("test", RelationalOperator::LessOrEqualTo, 5.2);
        assert!(c.is_met(5.2));
        assert!(!c.is_met(15.0));
        assert!(c.is_met(1.0));

        let c = Constraint::new_with_modifiers("test", RelationalOperator::EqualTo, 5.2, 1.0, -1.0);
        assert!(c.is_met(4.2));

        let c = Constraint::new_with_modifiers("test", RelationalOperator::EqualTo, 5.0, 0.5, 1.0);
        assert!(c.is_met(3.5));
    }

    #[test]
    fn test_constraint_violation() {
        let c = Constraint::new("test", RelationalOperator::EqualTo, 5.2);
        assert_eq!(c.constraint_violation(5.2), 0.0);
        assert_eq!(c.constraint_violation(1.2), 4.0);
        assert_eq!(c.constraint_violation(-1.2), 6.4);

        let c = Constraint::new("test", RelationalOperator::NotEqualTo, 5.2);
        assert_eq!(c.constraint_violation(5.2), 1.0);
        assert_eq!(c.constraint_violation(1.0), 0.0);

        let c = Constraint::new("test", RelationalOperator::LessThan, 5.2);
        assert_eq!(c.constraint_violation(0.0), 0.0);
        assert_approx_eq!(f64, c.constraint_violation(9.2), 4.0, epsilon = 0.001);

        let c = Constraint::new("test", RelationalOperator::GreaterThan, 5.2);
        assert_eq!(c.constraint_violation(10.0), 0.0);
        assert_approx_eq!(f64, c.constraint_violation(2.2), 3.0, epsilon = 0.001);

        let c = Constraint::new("test", RelationalOperator::LessOrEqualTo, 5.2);
        assert_eq!(c.constraint_violation(0.0), 0.0);
        assert_eq!(c.constraint_violation(5.2), 0.0);
        assert_approx_eq!(f64, c.constraint_violation(9.2), 4.0, epsilon = 0.001);

        let c = Constraint::new("test", RelationalOperator::GreaterOrEqualTo, 5.2);
        assert_eq!(c.constraint_violation(10.0), 0.0);
        assert_eq!(c.constraint_violation(5.2), 0.0);
        assert_approx_eq!(f64, c.constraint_violation(2.2), 3.0, epsilon = 0.001);
    }
}
