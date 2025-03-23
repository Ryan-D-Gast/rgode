//! Defines Solout Trait and Provides Common Solout Implementations

// Solout Trait for controlling output of the solver
mod solout;
pub use solout::Solout;

// Common Solout Implementations
mod default;
pub use default::DefaultSolout;

mod even;
pub use even::EvenSolout;

mod dense;
pub use dense::DenseSolout;

mod t_eval;
pub use t_eval::TEvalSolout;

// Crossing Detecting Solouts 

/// Defines the direction of threshold crossing to detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingDirection {
    /// Detect crossings in both directions
    Both,
    /// Detect only crossings from below to above the threshold (positive direction)
    Positive,
    /// Detect only crossings from above to below the threshold (negative direction)
    Negative,
}

// Crossing detection solout
mod crossing;
pub use crossing::CrossingSolout;

// Hyperplane crossing detection solout
mod hyperplane;
pub use hyperplane::HyperplaneCrossingSolout;