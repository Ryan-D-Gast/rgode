//! Solve IVP function

use nalgebra::SMatrix;

use crate::{EventAction, Solution, Solver, SolverStatus, System, Solout};
use crate::traits::{Real, EventData};
use std::time::Instant;

/// Solves an Initial Value Problem (IVP) for a system of ordinary differential equations.
/// 
/// This is the core solution function that drives the numerical integration of ODEs.
/// It handles initialization, time stepping, event detection, and solution output
/// according to the provided output strategy.
/// 
/// Note that it is recommend to use the `IVP` struct to solve the ODEs, 
/// as it provides far more feature rich and convenient interface which
/// wraps this function. See examples on github for more details.
/// 
/// # Overview
/// 
/// An Initial Value Problem takes the form:
/// 
/// ```text
/// dy/dt = f(t, y),  t ∈ [t0, tf],  y(t0) = y0
/// ```
/// 
/// This function solves such a problem by:
/// 
/// 1. Initializing the solver with the system and initial conditions
/// 2. Stepping the solver through the integration interval
/// 3. Detecting and handling events (if any)
/// 4. Collecting solution points according to the specified output strategy
/// 5. Monitoring for errors or exceptional conditions
/// 
/// # Arguments
/// 
/// * `solver` - Configured solver instance with appropriate settings (e.g., tolerances)
/// * `system` - The ODE system that implements the `System` trait
/// * `t0` - Initial time point
/// * `tf` - Final time point (can be less than `t0` for backward integration)
/// * `y0` - Initial state vector
/// * `solout` - Solution output strategy that controls which points are included in the result
/// 
/// # Returns
/// 
/// * `Ok(Solution)` - If integration completes successfully or is terminated by an event
/// * `Err(SolverStatus)` - If an error occurs (e.g., excessive stiffness, maximum steps reached)
/// 
/// # Solution Object
/// 
/// The returned `Solution` object contains:
/// 
/// * `t` - Vector of time points
/// * `y` - Vector of state vectors at each time point
/// * `status` - Final solver status (Complete or Interrupted)
/// * `evals` - Number of function evaluations performed
/// * `steps` - Total number of steps attempted
/// * `rejected_steps` - Number of steps rejected by the error control
/// * `accepted_steps` - Number of steps accepted by the error control
/// * `solve_time` - Wall time taken for the integration
/// 
/// # Event Handling
/// 
/// The solver checks for events after each step using the `event` method of the system.
/// If an event returns `EventAction::Terminate`, the integration stops and interpolates
/// to find the precise point where the event occurred, using a modified regula falsi method.
/// 
/// # Examples
/// 
/// ```
/// use rgode::prelude::*;
/// use rgode::solve_ivp; 
/// use rgode::solout::DefaultSolout;
/// use nalgebra::Vector1;
/// 
/// // Define a simple exponential growth system: dy/dt = y
/// struct ExponentialGrowth;
/// 
/// impl System<f64, 1, 1> for ExponentialGrowth {
///     fn diff(&self, _t: f64, y: &Vector1<f64>, dydt: &mut Vector1<f64>) {
///         dydt[0] = y[0];
///     }
/// }
/// 
/// // Solve from t=0 to t=1 with initial condition y=1
/// let mut solver = DOP853::new().rtol(1e-8).atol(1e-10);
/// let system = ExponentialGrowth;
/// let y0 = Vector1::new(1.0);
/// let result = solve_ivp(&mut solver, &system, 0.0, 1.0, &y0, DefaultSolout::new());
/// 
/// match result {
///     Ok(solution) => {
///         println!("Final value: {}", solution.y.last().unwrap()[0]);
///         println!("Number of steps: {}", solution.steps);
///     },
///     Err(status) => {
///         println!("Integration failed: {:?}", status);
///     }
/// }
/// ```
/// 
/// # Notes
/// 
/// * For forward integration, `tf` should be greater than `t0`.
/// * For backward integration, `tf` should be less than `t0`.
/// * The `tf == t0` case is considered an error (no integration to perform).
/// * The output points depend on the chosen `Solout` implementation.
/// 
pub fn solve_ivp<T, const R: usize, const C: usize, E, S, F, O>(solver: &mut S, system: &F, t0: T, tf: T, y0: &SMatrix<T, R, C>, mut solout: O) -> Result<Solution<T, R, C, E, O>, SolverStatus<T, R, C, E>>
where 
    T: Real,
    E: EventData,
    F: System<T, R, C, E>,
    S: Solver<T, R, C, E>,
    O: Solout<T, R, C, E>
{
    // Timer for measuring solve time
    let start = Instant::now();

    // Determine integration direction and check that tf != t0
    let integration_direction =  match (tf - t0).signum() {
        x if x == T::one() => T::one(),
        x if x == T::from_f64(-1.0).unwrap() => T::from_f64(-1.0).unwrap(),
        _ => return Err(SolverStatus::BadInput("Final time tf must be different from initial time t0.".to_string())),
    };

    // Clear statistics in case it was used before and reset solver and check for errors
    match solver.init(system, t0, tf, &y0) {
        Ok(_) => {}
        Err(e) => return Err(e),
    }

    // Solution Vectors
    let mut t_out: Vec<T> = Vec::with_capacity(100); // Pre-allocate space for 100 time points
    let mut y_out: Vec<SMatrix<T, R, C>> = Vec::with_capacity(100);

    // Add initial point to output if include_t0_tf is true
    if solout.include_t0_tf() {
        t_out.push(t0);
        y_out.push(y0.clone());
    }

    // For event
    let mut tc: T = t0;
    let mut ts: T;

    // Check Terminate before starting incase the initial conditions trigger it
    match system.event(t0, &y0, solver.dydt()) {
        EventAction::Continue => {}
        EventAction::Terminate(reason) => {
            return Ok(Solution {
                y: y_out,
                t: t_out,
                solout: solout,
                status: SolverStatus::Interrupted(reason),
                evals: solver.evals(),
                steps: solver.steps(),
                rejected_steps: solver.rejected_steps(),
                accepted_steps: solver.steps(),
                solve_time: T::from_f64(start.elapsed().as_secs_f64()).unwrap(),
            });
        }
    }

    // Set Solver to Solving
    solver.set_status(SolverStatus::Solving);

    // Main Loop
    let mut solving = true;
    while solving {
        // Check if next step overshoots tf
        if (solver.t() + solver.h() - tf) * integration_direction > T::zero() {
            // Correct step size to reach tf
            solver.set_h(tf - solver.t());
            solving = false;
        }

        // Check if Step Size is too smaller then machine default_epsilon
        if solver.h().abs() < T::default_epsilon() {
            // If the step size is converging at the final point then consider it solved
            if (solver.t() - tf).abs() < T::default_epsilon() {
                break;
            // Otherwise, return StepSize error
            } else {
                solver.set_status(SolverStatus::StepSize(solver.t(), *solver.y()));
                break;
            }
        }

        // Perform a step
        solver.step(system);

        // Check for rejected step
        match solver.status() {
            SolverStatus::Solving => {}
            SolverStatus::RejectedStep => continue,
            _ => break,
        }
        
        // Record the result
        solout.solout(solver, system, &mut t_out, &mut y_out);

        // Check event condition
        match system.event(solver.t(), solver.y(), solver.dydt()) {
            EventAction::Continue => { 
                // Update last continue point
                tc = solver.t(); 
            }
            EventAction::Terminate(re) => {
                // For iteration to event point
                let mut reason = re;

                // Update last stop point
                ts = solver.t();

                // If event_tolerance is set, interpolate to the point where event is triggered
                // Method: Regula Falsi (False Position) with Illinois adjustment
                if let Some(tol) = system.event_tolerance() {
                    let mut side_count = 0;   // Illinois method counter
                    
                    // For Illinois method adjustment
                    let mut f_low: T = T::from_f64(-1.0).unwrap();     // Continue represented as -1
                    let mut f_high: T = T::from_f64(1.0).unwrap();     // Terminate represented as +1
                    let mut t_guess: T;
                    
                    let max_iterations = 20; // Prevent infinite loops
                    let mut dydt = SMatrix::<T, R, C>::zeros(); // Derivative vector
                    
                    // False position method with Illinois adjustment
                    for _ in 0..max_iterations {
                        // Check if we've reached desired precision
                        if (ts - tc).abs() <= tol {
                            break;
                        }
                        
                        // False position formula with Illinois adjustment
                        t_guess = (tc * f_high - ts * f_low) / (f_high - f_low);
                        
                        // Protect against numerical issues
                        if !t_guess.is_finite() || 
                            (integration_direction > T::zero() && (t_guess <= tc || t_guess >= ts)) ||
                            (integration_direction < T::zero() && (t_guess >= tc || t_guess <= ts)) {
                            t_guess = (tc + ts) / T::from_f64(2.0).unwrap();  // Fall back to bisection
                        }
                        
                        // Interpolate state at guess point
                        let y = solver.interpolate(system, t_guess);
                        
                        // Check event at guess point
                        system.diff(t_guess, &y, &mut dydt);
                        match system.event(t_guess, &y, &dydt) {
                            EventAction::Continue => {
                                tc = t_guess;

                                // Illinois adjustment to improve convergence
                                side_count += 1;
                                if side_count >= 2 {
                                    f_high /= T::from_f64(2.0).unwrap();  // Reduce influence of high point
                                    side_count = 0;
                                }
                            }
                            EventAction::Terminate(re) => {
                                reason = re;
                                ts = t_guess;
                                side_count = 0;
                                f_low = T::from_f64(-1.0).unwrap();  // Reset low point influence
                            }
                        }
                    }
                }

                // Final event point
                let y_final = solver.interpolate(system, ts);
                
                // Remove points after the event point and add the event point
                // Find the cutoff index based on integration direction
                let cutoff_index = if integration_direction > T::zero() {
                    // Forward integration - find first index where t > ts
                    t_out.iter().position(|&t| t > ts)
                } else {
                    // Backward integration - find first index where t < ts
                    t_out.iter().position(|&t| t < ts)
                };

                // If we found a cutoff point, truncate both vectors
                if let Some(idx) = cutoff_index {
                    t_out.truncate(idx);
                    y_out.truncate(idx);
                }

                // Add the event point
                t_out.push(ts);
                y_out.push(y_final);

                return Ok(Solution {
                    y: y_out,
                    t: t_out,
                    solout: solout,
                    status: SolverStatus::Interrupted(reason),
                    evals: solver.evals(),
                    steps: solver.steps(),
                    rejected_steps: solver.rejected_steps(),
                    accepted_steps: solver.steps(),
                    solve_time: T::from_f64(start.elapsed().as_secs_f64()).unwrap(),
                });
            }
        }
    }

    // Check for problems in Solver Status
    match solver.status() {
        SolverStatus::Solving => {
            solver.set_status(SolverStatus::Complete);

            // Add final point to output if include_t0_tf is true
            if solout.include_t0_tf() && t_out.last().copied() != Some(tf) {
                t_out.push(tf);
                y_out.push(solver.y().clone());
            }

            Ok(Solution {
                y: y_out,
                t: t_out,
                solout: solout,
                status: solver.status().clone(),
                evals: solver.evals(),
                steps: solver.steps(),
                rejected_steps: solver.rejected_steps(),
                accepted_steps: solver.steps(),
                solve_time: T::from_f64(start.elapsed().as_secs_f64()).unwrap(),
            })
        }
        status => Err(status.clone()),
    }
}