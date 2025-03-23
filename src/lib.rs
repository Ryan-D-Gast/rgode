//! # rgode
//! 
//! `rgode` is a Rust library for solving ordinary differential equations (ODEs) using various numerical methods.
//! 
//! GitHub: [rgode](https://github.com/Ryan-D-Gast/rgode)
//! 
//! The library is designed around an initial value problem (IVP) struct which contains 
//! a system struct that implements the `System` trait. The `System` traits implements the
//! functions
//! - `diff`: Differential Equation dydt = f(t, y) in form f(t, &y, &mut dydt).
//! - `event`: Optional event function to interrupt solver when condition is met or event occurs.
//! 
//! In addition a `Solout` trait is used to control the output of the solver.
//! Numerous Solout implementations are provided in the `solout` module.
//! These are used by the `IVP` struct via methods like `even`, `dense`, and `t_eval`.
//! 
//! Solvers are then selected and initialized with the desired settings and the IVP is solved
//! via `ivp.solve(&mut solver)`. The solution is then returned as a `Solution` struct.
//! 
//! `rgode` contains numerous fixed steps and adaptive step methods for solving ODEs.
//! On github read the README.md and view the examples for usage of more advanced features.
//! 
//! ## Feature Flags
//! 
//! - `polars`: Enable converting Solution to Polars DataFrame using `Solution.to_polars()`.
//! 
//! ## Example
//! 
//!```rust
//! use rgode::prelude::*;
//! use nalgebra::{SVector, vector};
//! 
//! pub struct LinearEquation {
//!     pub a: f64,
//!     pub b: f64,
//! }
//! 
//! impl System<f64, 1, 1> for LinearEquation {
//!     fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
//!         dydt[0] = self.a + self.b * y[0];
//!     }
//! }
//! 
//! fn main() {
//!     let system = LinearEquation { a: 1.0, b: 2.0 };
//!     let t0 = 0.0;
//!     let tf = 1.0;
//!     let y0 = vector![1.0];
//!     let ivp = IVP::new(system, t0, tf, y0);
//!     let mut solver = DOP853::new().rtol(1e-8).atol(1e-6);
//!     let solution = match ivp.solve(&mut solver) {
//!         Ok(sol) => sol,
//!         Err(e) => panic!("Error: {:?}", e),
//!     };
//! 
//!     for (t, y) in solution.iter() {
//!        println!("t: {:.4}, y: {:.4}", t, y[0]);
//!     }
//! }
//!```
//! 
//! # License
//! 
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

// Prelude module recommended users call this e.g. `use rgode::prelude::*;` to import commonly used types and traits.
pub mod prelude;

// Float and Vector traits
pub mod traits;

// IVP Struct which is used to solve the ODE given the system and a solver
mod ivp;
pub use ivp::{
    IVP,
    solve_ivp, 
};

// System Trait for Differential Equations
mod system;
pub use system::{
    System,       // System Trait for Differential Equations
    EventAction,  // Command to the Solver for Returned in Terminate Function in System Trait
};

// Solver Traits for ODE Solvers.
mod solver;
pub use solver::{
    Solver,             // Solver Trait for ODE Solvers
    SolverStatus,       // Status of the Solver for Control Flow and Error Handling
};

// Solout Trait for controlling output of the solver
pub mod solout; // Numerous implementations of the Solout trait are contained in this module
pub use solout::{
    // Solout Trait for controlling output of the solver
    Solout,
    CrossingDirection,
};

// Solution of a solved IVP Problem
mod solution;
pub use solution::Solution;

// Solver for ODEs
pub mod solvers;

// Interpolation Functions
pub mod interpolate;