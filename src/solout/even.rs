//! Even Solout Implementation for evenly spaced output points.
//!
//! This module provides an output strategy that generates solution points at evenly
//! spaced time intervals, regardless of the actual steps taken by the solver.

use nalgebra::SMatrix;

use crate::{Solver, Solout, System};
use crate::traits::{Real, EventData};

/// An output handler that provides solution points at evenly spaced time intervals.
/// 
/// # Overview
///
/// `EvenSolout` generates output points at strictly uniform time intervals, creating
/// a regular grid of solution points. This is especially useful for visualization,
/// post-processing, or when interfacing with other systems that expect uniformly
/// sampled data.
///
/// Unlike `DefaultSolout` which captures the naturally occurring solver steps (which
/// may have varying time intervals), `EvenSolout` uses interpolation to evaluate the
/// solution at precise time points separated by a fixed interval.
///
/// # Example
///
/// ```
/// use rgode::prelude::*;
/// use rgode::solout::EvenSolout;
/// use nalgebra::{Vector2, vector};
///
/// // Simple harmonic oscillator
/// struct HarmonicOscillator;
///
/// impl System<f64, 2, 1> for HarmonicOscillator {
///     fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
///         // y[0] = position, y[1] = velocity
///         dydt[0] = y[1];
///         dydt[1] = -y[0];
///     }
/// }
///
/// // Create the system and solver
/// let system = HarmonicOscillator;
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![1.0, 0.0];
/// let mut solver = DOP853::new().rtol(1e-6).atol(1e-8);
///
/// // Generate output points with a fixed interval of 0.1
/// let even_output = EvenSolout::new(0.1, t0, tf);
///
/// // Solve with evenly spaced output
/// let ivp = IVP::new(system, t0, tf, y0);
/// let solution = ivp.solout(even_output).solve(&mut solver).unwrap();
///
/// // Note: This is equivalent to using the convenience method:
/// let solution = ivp.even(0.1).solve(&mut solver).unwrap();
/// ```
///
/// # Output Characteristics
///
/// The output will contain points at regular intervals: t₀, t₀+dt, t₀+2dt, ..., tₙ.
/// The final point tₙ is guaranteed to be included, even if it doesn't fall exactly on
/// the regular grid. Any evaluation points that fall outside the integration range are ignored.
///
pub struct EvenSolout<T: Real> {
    /// Fixed time interval between points
    dt: T,
    /// Initial time to align intervals with
    t0: T,
    /// Final time to ensure the last point is included
    tf: T,
    /// Direction of integration (positive for forward, negative for backward)
    direction: T,
    /// Last time point that was output
    last_output_t: Option<T>,
}

impl<T, const R: usize, const C: usize, E> Solout<T, R, C, E> for EvenSolout<T>
where 
    T: Real,
    E: EventData
{
    fn solout<S, F>(&mut self, solver: &mut S, system: &F, t_out: &mut Vec<T>, y_out: &mut Vec<SMatrix<T, R, C>>)
    where 
        F: System<T, R, C, E>,
        S: Solver<T, R, C, E>
    {
        let t_curr = solver.t();
        let t_prev = solver.t_prev();
        
        // Determine the alignment offset (remainder when divided by dt)
        let offset = self.t0 % self.dt;
        
        // Start from the last output point if available, otherwise from t_prev
        let start_t = match self.last_output_t {
            Some(t) => t + self.dt * self.direction,
            None => {
                // First time through, we need to include t0
                if (t_prev - self.t0).abs() < T::default_epsilon() {
                    t_out.push(self.t0);
                    y_out.push(solver.y_prev().clone());
                    self.last_output_t = Some(self.t0);
                    self.t0 + self.dt * self.direction
                } else {
                    // Find the next aligned point after t_prev
                    let rem = (t_prev - offset) % self.dt;
                    
                    if self.direction > T::zero() {
                        // For forward integration
                        if rem.abs() < T::default_epsilon() {
                            t_prev
                        } else {
                            t_prev + (self.dt - rem)
                        }
                    } else {
                        // For backward integration
                        if rem.abs() < T::default_epsilon() {
                            t_prev
                        } else {
                            t_prev - rem
                        }
                    }
                }
            }
        };
        
        let mut ti = start_t;
        
        // Interpolate between steps
        while (self.direction > T::zero() && ti <= t_curr) || 
              (self.direction < T::zero() && ti >= t_curr) {
            // Only output if the point falls within the current step
            if (self.direction > T::zero() && ti >= t_prev && ti <= t_curr) ||
               (self.direction < T::zero() && ti <= t_prev && ti >= t_curr) {
                let yi = solver.interpolate(system, ti);
                t_out.push(ti);
                y_out.push(yi);
                self.last_output_t = Some(ti);
            }
            
            // Move to the next point
            ti = ti + self.dt * self.direction;
        }
        
        // Include final point if this step reaches tf and we haven't added it yet
        if t_curr == self.tf && (self.last_output_t.is_none() || self.last_output_t.unwrap() != self.tf) {
            t_out.push(self.tf);
            y_out.push(solver.y().clone());
            self.last_output_t = Some(self.tf);
        }
    }
}

impl<T: Real> EvenSolout<T> {
    /// Creates a new EvenSolout instance with the specified time interval.
    ///
    /// This output handler will generate solution points at regular intervals of `dt`
    /// starting from `t0` and continuing through `tf`. The points will be aligned
    /// with `t0` (i.e., t₀, t₀+dt, t₀+2dt, ...).
    ///
    /// # Arguments
    /// * `dt` - The fixed time interval between output points
    /// * `t0` - The initial time of the integration
    /// * `tf` - The final time of the integration
    ///
    /// # Returns
    /// * A new `EvenSolout` instance
    ///
    pub fn new(dt: T, t0: T, tf: T) -> Self {
        EvenSolout { 
            dt, 
            t0, 
            tf,
            direction: (tf - t0).signum(),
            last_output_t: None,
        }
    }
}