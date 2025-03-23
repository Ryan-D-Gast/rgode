use rgode::prelude::*;
use nalgebra::{SVector, vector};

/// Damped Harmonic Oscillator System
struct DampedOscillator {
    damping: f64, // Damping coefficient
    spring_constant: f64, // Spring constant
}

impl System<f64, 2> for DampedOscillator {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        // Pure function, no state updates
        dydt[0] = y[1];
        dydt[1] = -self.damping * y[1] - self.spring_constant * y[0];
    }
}

fn main() {
    // Initialize the solver
    let mut solver = DOPRI5::new(0.01).rtol(1e-12).atol(1e-12);

    // Define the system parameters
    let damping = 0.5;
    let spring_constant = 1.0;
    let system = DampedOscillator {
        damping,
        spring_constant,
    };

    // Define the initial conditions
    let y0 = vector![1.0, 0.0]; // Initial position and velocity
    let t0 = 0.0;
    let tf = 20.0;

    // Create the IVP
    let damped_oscillator_ivp = IVP::new(system, t0, tf, y0);

    // Solve the IVP
    match damped_oscillator_ivp
        .crossing(0, 0.0, CrossingDirection::Both)
        .solve(&mut solver) {
        Ok(solution) => {
            println!("Solution:");
            println!("Time, Position, Velocity");
            for (t, y) in solution.iter() {
                println!("{:.4}, {:.4}, {:.4}", t, y[0], y[1]);
            }

            println!("Function evaluations: {}", solution.evals);
            println!("Steps: {}", solution.steps);
            println!("Rejected Steps: {}", solution.rejected_steps);
            println!("Accepted Steps: {}", solution.accepted_steps);
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}