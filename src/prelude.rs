//! Prelude module that exports most commonly used items
//!
//! This module provides the most commonly used types, traits, and functions
//! from rgode so they can be imported with a single `use` statement:
//!
//! ```
//! use rgode::prelude::*;
//! ```
//! 
//! # Includes
//! 
//! ## Core API
//! * `IVP` struct for defining initial value problems
//! * `Solution` struct for storing the results of a solved IVP
//! * `SolverStatus` enum for checking the status of the solver
//! 
//! ## Defining systems and solout
//! * `System` trait for defining system of differential equations
//! * `EventAction` return enum for system.event function
//! * `Solout` trait for controlling output of the solver
//! 
//! ## Solvers
//! * `RK4` Classic 4th-order Runge-Kutta method
//! * `DOP853` Dormand-Prince 8th-order method
//! * `DOPRI5` Dormand-Prince 5th-order method
//! * `APCF4` Fixed step 4th-order Adams Predictor-Corrector method
//! * `APCV4` Adaptive step 4th-order Adams Predictor-Corrector method
//! 
//! Note more solvers are available in the `solvers` module.
//! 
//! ## Solout
//! * `DefaultSolout` for capturing all solver steps
//! * `EvenSolout` for capturing evenly spaced solution points
//! * `DenseSolout` for capturing a dense set of interpolated points
//! * `TEvalSolout` for capturing points based on a user-defined function
//! * `CrossingSolout` for capturing points when crossing a specified value
//! * `HyperplaneCrossingSolout` for capturing points when crossing a hyperplane
//! 
//! Note more solout options are available in the `solout` module.
//! 
//! # Miscellaneous traits to expose API
//! * `Solver` trait for defining solvers, not used by users but here so trait methods can be used.
//! 
//! # License
//! ```text
//! Copyright 2025 Ryan D. Gast
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.
//! ```
//! 

// Core structs and traits
pub use crate::{
    // Core types
    IVP,
    Solution,
    
    // Core traits
    System,
    Solver,
    
    // Control flow
    EventAction,
    SolverStatus,
};

// Re-export nalgebra types
pub use nalgebra::{SVector, vector, SMatrix, matrix};

// Output handlers
pub use crate::solout::{
    Solout,
    DefaultSolout,
    DenseSolout,
    EvenSolout,
    TEvalSolout,
    CrossingDirection,
    CrossingSolout,
    HyperplaneCrossingSolout,
};

// Common solvers
pub use crate::solvers::{
    // Popular fixed step methods
    runge_kutta::RK4,
    
    // Popular adaptive methods
    runge_kutta::DOP853,
    runge_kutta::DOPRI5,
    
    // Adams Predictor Correct methods
    adams::APCF4,
    adams::APCV4,
};