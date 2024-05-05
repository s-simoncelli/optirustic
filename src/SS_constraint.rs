// use std::fmt::{Debug, Formatter, Write};
//
// /// Operator used to check a bounded constraint
// pub enum RelationalOperator {
//     /// Value must equal the constraint value
//     EqualTo,
//     /// Value must not equal the constraint value
//     NotEqualTo,
//     /// Value must be less or equal to the constraint value
//     LessOrEqualTo,
//     /// Value must be less than the constraint value
//     LessThan,
//     /// Value must be greater or equal to the constraint value
//     GreaterOrEqualTo,
//     /// Value must be greater than the constraint value
//     GreaterThan,
// }
//
// ///A trait to implement for a constraint
// pub trait BaseConstraint {
//     /// Get the constraint name.
//     fn name(&self) -> String;
//     /// Check whether the constraint is met.
//     fn is_met(&self, value: f64) -> bool;
// }
//
// // TODO add equality with eps (see pymoo)
// // TODO hwo to check that value type matches target (can be float or int?)
//
// /// Define a constraint where a value is compared with a relational operator as follows:
// ///  - Equality operator (`RelationalOperator::EqualTo`): value == target
// ///  - Inequality operator (`RelationalOperator::NotEqualTo`): value != target
// ///  - Greater than operator (`RelationalOperator::GreaterThan`): value > target
// ///  - Greater or equal to operator (`RelationalOperator::GreaterOrEqualTo`): value >= target
// ///  - less than operator (`RelationalOperator::LessThan`): value < target
// ///  - less or equal to operator (`RelationalOperator::LessOrEqualTo`): value <= target
// ///
// /// # Example
// ///
// /// ```
// ///   let c = RelationalConstraint::new(RelationalOperator::GreaterOrEqualTo, 5.2);
// ///   assert_eq!(c.is_met(10.1), true);
// ///   assert_eq!(c.is_met(3.11), false);
// /// ```
// pub struct RelationalConstraint {
//     /// The constraint name.
//     name: String,
//     /// The relational operator to use to compare a value against the constraint target value.
//     operator: RelationalOperator,
//     /// The constraint target.
//     target: f64,
// }
//
// impl RelationalConstraint {
//     /// Create a new relational constraint.
//     ///
//     /// # Arguments
//     ///
//     /// * `name`: The constraint name.
//     /// * `operator`: The relational operator to use to compare a value against the constraint
//     /// target value.
//     /// * `target`: The constraint target.
//     ///
//     /// returns: `RelationalConstraint`
//     fn new(name: &str, operator: RelationalOperator, target: f64) -> Self {
//         Self {
//             name: name.to_owned(),
//             operator,
//             target,
//         }
//     }
// }
//
// impl Debug for RelationalConstraint {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let sign = match self.operator {
//             RelationalOperator::EqualTo => "==",
//             RelationalOperator::NotEqualTo => "!=",
//             RelationalOperator::LessOrEqualTo => "<=",
//             RelationalOperator::LessThan => "<",
//             RelationalOperator::GreaterOrEqualTo => ">=",
//             RelationalOperator::GreaterThan => ">",
//         };
//         f.write_fmt(format_args!("{}{}", sign, self.target))
//     }
// }
//
// impl BaseConstraint for RelationalConstraint {
//     fn name(&self) -> String {
//         self.name.clone()
//     }
//
//     fn is_met(&self, value: f64) -> bool {
//         match self.operator {
//             RelationalOperator::EqualTo => value == self.target,
//             RelationalOperator::NotEqualTo => value != self.target,
//             RelationalOperator::LessOrEqualTo => value <= self.target,
//             RelationalOperator::LessThan => value < self.target,
//             RelationalOperator::GreaterOrEqualTo => value >= self.target,
//             RelationalOperator::GreaterThan => value > self.target,
//         }
//     }
// }
//
// pub struct ConstraintWithBound {
//     /// The constraint name.
//     name: String,
//     /// The constraint lower bound.
//     lower_bound: f64,
//     /// The constraint lower bound.
//     upper_bound: f64,
//     /// Whether the value must be strictly larger than the lower bound.
//     include_lower_bound: bool,
//     /// Whether the value must be strictly smaller than the upper bound.
//     include_upper_bound: bool,
// }
//
// impl ConstraintWithBound {
//     /// Check that a value is within an upper and lower limit.
//     ///
//     /// # Arguments
//     ///
//     /// * `name`: The constraint name.
//     /// * `lower_bound`: The constraint lower bound.
//     /// * `upper_bound`: The constraint upper bound.
//     /// * `include_lower_bound`: If false, the value must be strictly larger than the `lower_bound`.
//     /// Default to true.
//     /// * `include_upper_bound`: If false, the value must be strictly smaller than the `upper_bound`.
//     //  Dfault to true.
//     ///
//     /// returns: `ConstraintWithBound`
//     fn new(
//         name: &str,
//         lower_bound: f64,
//         upper_bound: f64,
//         include_lower_bound: Option<bool>,
//         include_upper_bound: Option<bool>,
//     ) -> Self {
//         Self {
//             name: name.to_owned(),
//             lower_bound,
//             upper_bound,
//             include_lower_bound: include_lower_bound.unwrap_or(false),
//             include_upper_bound: include_upper_bound.unwrap_or(false),
//         }
//     }
// }
//
// impl Debug for ConstraintWithBound {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let (bracket1, bracket2) = if self.include_lower_bound && self.include_upper_bound {
//             ("[", "]")
//         } else if !self.include_lower_bound && self.include_upper_bound {
//             ("]", "]")
//         } else if self.include_lower_bound && !self.include_upper_bound {
//             ("[", "[")
//         } else {
//             ("[", "]")
//         };
//         f.write_fmt(format_args!(
//             "{}{}; {}{}",
//             bracket1, self.lower_bound, self.upper_bound, bracket2
//         ))
//     }
// }
//
// impl BaseConstraint for ConstraintWithBound {
//     fn name(&self) -> String {
//         self.name.clone()
//     }
//
//     fn is_met(&self, value: f64) -> bool {
//         if self.include_lower_bound && self.include_upper_bound {
//             (value >= self.lower_bound) & (value <= self.upper_bound)
//         } else if !self.include_lower_bound && self.include_upper_bound {
//             (value > self.lower_bound) & (value <= self.upper_bound)
//         } else if self.include_lower_bound && !self.include_upper_bound {
//             (value >= self.lower_bound) & (value < self.upper_bound)
//         } else {
//             (value > self.lower_bound) & (value < self.upper_bound)
//         }
//     }
// }
//
// #[derive(Debug)]
// pub enum Constraint {
//     RelationalConstraint(RelationalConstraint),
//     ConstraintWithBound(ConstraintWithBound),
// }
//
// impl Constraint {
//     /// Get the constraint name.
//     pub fn name(&self) -> String {
//         match &self {
//             Constraint::RelationalConstraint(c) => c.name(),
//             Constraint::ConstraintWithBound(c) => c.name(),
//         }
//     }
//
//     /// Check whether the constraint is met.
//     ///
//     /// # Arguments
//     ///
//     /// * `value`: The value to check against the constraint value(s).
//     ///
//     /// returns: `bool`
//     pub fn is_met(&self, value: f64) -> bool {
//         match &self {
//             Constraint::RelationalConstraint(c) => c.is_met(value),
//             Constraint::ConstraintWithBound(c) => c.is_met(value),
//         }
//     }
// }
//
// #[cfg(test)]
// mod test {
//     use crate::constraint::{
//         BaseConstraint, ConstraintWithBound, RelationalConstraint, RelationalOperator,
//     };
//
//     #[test]
//     fn test_relational_constraints() {
//         let c = RelationalConstraint::new("test", RelationalOperator::EqualTo, 5.2);
//         assert!(c.is_met(5.2));
//         assert!(!c.is_met(15.0));
//
//         let c = RelationalConstraint::new("test", RelationalOperator::NotEqualTo, 5.2);
//         assert!(!c.is_met(5.2));
//         assert!(c.is_met(15.0));
//
//         let c = RelationalConstraint::new("test", RelationalOperator::GreaterThan, 5.2);
//         assert!(!c.is_met(5.2));
//         assert!(c.is_met(15.0));
//         assert!(!c.is_met(1.0));
//
//         let c = RelationalConstraint::new("test", RelationalOperator::GreaterOrEqualTo, 5.2);
//         assert!(c.is_met(5.2));
//         assert!(c.is_met(15.0));
//         assert!(!c.is_met(1.0));
//
//         let c = RelationalConstraint::new("test", RelationalOperator::LessThan, 5.2);
//         assert!(!c.is_met(5.2));
//         assert!(c.is_met(1.0));
//         assert!(!c.is_met(15.0));
//
//         let c = RelationalConstraint::new("test", RelationalOperator::LessOrEqualTo, 5.2);
//         assert!(c.is_met(5.2));
//         assert!(!c.is_met(15.0));
//         assert!(c.is_met(1.0));
//     }
//
//     #[test]
//     fn test_boundary_constraints() {
//         let c = ConstraintWithBound::new("test", 1.0, 5.0, Some(true), Some(true));
//         assert!(c.is_met(2.0));
//         assert!(c.is_met(1.0));
//         assert!(c.is_met(5.0));
//         assert!(!c.is_met(-5.0));
//         assert!(!c.is_met(15.0));
//
//         let c = ConstraintWithBound::new("test", 1.0, 5.0, Some(false), Some(true));
//         assert!(c.is_met(2.0));
//         assert!(!c.is_met(1.0));
//         assert!(c.is_met(5.0));
//         assert!(!c.is_met(-5.0));
//         assert!(!c.is_met(15.0));
//
//         let c = ConstraintWithBound::new("test", 1.0, 5.0, Some(true), Some(false));
//         assert!(c.is_met(2.0));
//         assert!(c.is_met(1.0));
//         assert!(!c.is_met(5.0));
//         assert!(!c.is_met(-5.0));
//         assert!(!c.is_met(15.0));
//
//         let c = ConstraintWithBound::new("test", 1.0, 5.0, Some(false), Some(false));
//         assert!(c.is_met(2.0));
//         assert!(!c.is_met(1.0));
//         assert!(!c.is_met(5.0));
//         assert!(!c.is_met(-5.0));
//         assert!(!c.is_met(15.0));
//     }
// }
