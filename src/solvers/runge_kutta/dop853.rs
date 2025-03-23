//! DOP853 Solver for Ordinary Differential Equations.


use crate::{Solver, SolverStatus, System};
use crate::traits::{Real, EventData};
use crate::solvers::utils::{constrain_step_size, validate_step_size_parameters};
use nalgebra::SMatrix;

/// Dormand Prince 8(5, 3) Method for solving ordinary differential equations.
/// 8th order Dormand Prince method with 5th order error estimation and 3rd order interpolation.
/// 
/// Builds should begin with weight, normal, dense, or even methods.
/// and then chain the other methods to set the parameters.
/// The defaults should be great for most cases.
/// 
/// # Example
/// ```
/// use rgode::prelude::*;
/// 
/// let mut dop853 = DOP853::new()
///    .rtol(1e-12)
///    .atol(1e-12);
/// 
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![1.0, 0.0];
/// struct Example;
/// impl System<f64, 2, 1> for Example {
///    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
///       dydt[0] = y[1];
///       dydt[1] = -y[0];
///   }
/// }
/// let solution = IVP::new(Example, t0, tf, y0).solve(&mut dop853).unwrap();
/// 
/// let (t, y) = solution.last().unwrap();
/// println!("Solution: ({}, {})", t, y);
/// ```
/// 
/// # Settings
/// * `rtol`   - Relative tolerance for the solver.
/// * `atol`   - Absolute tolerance for the solver.
/// * `h0`     - Initial step size.
/// * `h_max`   - Maximum step size for the solver.
/// * `max_steps` - Maximum number of steps for the solver.
/// * `n_stiff` - Number of steps to check for stiffness.
/// * `safe`   - Safety factor for step size prediction.
/// * `fac1`   - Parameter for step size selection.
/// * `fac2`   - Parameter for step size selection.
/// * `beta`   - Beta for stabilized step size control.
/// 
/// # Default Settings
/// * `rtol`   - 1e-3
/// * `atol`   - 1e-6
/// * `h0`     - None (Calculated by solver if None)
/// * `h_max`   - None (Calculated by tf - t0 if None)
/// * `h_min`   - 0.0
/// * `max_steps` - 1_000_000
/// * `n_stiff` - 100
/// * `safe`   - 0.9
/// * `fac1`   - 0.33
/// * `fac2`   - 6.0
/// * `beta`   - 0.0
/// 
pub struct DOP853<T: Real, const R: usize, const C: usize, E: EventData> {
    // Initial Conditions
    pub h0: T,                // Initial Step Size

    // Final Time to Solve to
    tf: T,

    // Current iteration
    t: T,
    y: SMatrix<T, R, C>,
    h: T,

    // Tolerances 
    pub rtol: T,
    pub atol: T,

    // Settings
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
    pub n_stiff: usize,

    // DOP853 Specific Settings
    pub safe: T,
    pub fac1: T,
    pub fac2: T,
    pub beta: T,

    // Derived Settings
    expo1: T,
    facc1: T,
    facc2: T,
    facold: T,
    fac11: T,
    fac: T,

    // Iteration Tracking
    status: SolverStatus<T, R, C, E>,

    // Stiffness Detection
    h_lamb: T,
    non_stiff_counter: usize,
    stiffness_counter: usize,

    // Statistics
    evals: usize,
    steps: usize,
    rejected_steps: usize,
    accepted_steps: usize,

    // Derivatives
    k1: SMatrix<T, R, C>,    // Derivative at t
    k2: SMatrix<T, R, C>,
    k3: SMatrix<T, R, C>,
    k4: SMatrix<T, R, C>,
    k5: SMatrix<T, R, C>,
    k6: SMatrix<T, R, C>,
    k7: SMatrix<T, R, C>,
    k8: SMatrix<T, R, C>,
    k9: SMatrix<T, R, C>,
    k10: SMatrix<T, R, C>,

    // For Interpolation 
    cached_step_num: usize,
    y_old: SMatrix<T, R, C>, // State at Previous Step
    k_old: SMatrix<T, R, C>, // Derivative at Previous Step
    t_old: T, // Time of Previous Step
    h_old: T, // Step Size of Previous Step
    cont1: SMatrix<T, R, C>,
    cont2: SMatrix<T, R, C>,
    cont3: SMatrix<T, R, C>,
    cont4: SMatrix<T, R, C>,
    cont5: SMatrix<T, R, C>,
    cont6: SMatrix<T, R, C>,
    cont7: SMatrix<T, R, C>,
    cont8: SMatrix<T, R, C>,

    // Butcher Tableau Constants
    a21: T,
    a31: T,
    a32: T, 
    a41: T, 
    a43: T, 
    a51: T, 
    a53: T, 
    a54: T, 
    a61: T, 
    a64: T, 
    a65: T, 
    a71: T, 
    a74: T, 
    a75: T, 
    a76: T, 

    a81: T, 
    a84: T, 
    a85: T, 
    a86: T, 
    a87: T, 
    a91: T, 
    a94: T, 
    a95: T, 
    a96: T, 
    a97: T, 
    a98: T, 
    a101: T,
    a104: T,
    a105: T,
    a106: T,
    a107: T,
    a108: T,
    a109: T,

    a111: T,
    a114: T,
    a115: T,
    a116: T,
    a117: T,
    a118: T,
    a119: T,
    a1110: T, 
    a121: T,
    a124: T,
    a125: T,
    a126: T,
    a127: T,
    a128: T,
    a129: T,
    a1210: T, 
    a1211: T, 

    a141: T,
    a147: T,
    a148: T,
    a149: T,
    a1410: T, 
    a1411: T, 
    a1412: T, 
    a1413: T, 

    a151: T,
    a156: T,
    a157: T,
    a158: T,
    a1511: T, 
    a1512: T, 
    a1513: T, 
    a1514: T, 
    a161: T,
    a166: T,
    a167: T,
    a168: T,
    a169: T,
    a1613: T, 
    a1614: T, 
    a1615: T, 

    b1: T,
    b6: T,
    b7: T,
    b8: T,
    b9: T,
    b10: T, 
    b11: T, 
    b12: T, 

    bhh1: T,
    bhh2: T,
    bhh3: T,

    c2: T,
    c3: T,
    c4: T,
    c5: T,
    c6: T,
    c7: T,
    c8: T,
    c9: T,
    c10: T, 
    c11: T, 
    c14: T, 
    c15: T, 
    c16: T, 

    er1: T, 
    er6: T, 
    er7: T, 
    er8: T, 
    er9: T, 
    er10: T,
    er11: T,
    er12: T,

    d41: T, 
    d46: T, 
    d47: T, 
    d48: T, 
    d49: T, 
    d410: T,
    d411: T,
    d412: T,
    d413: T,
    d414: T,
    d415: T,
    d416: T,

    d51: T, 
    d56: T, 
    d57: T, 
    d58: T, 
    d59: T, 
    d510: T,
    d511: T,
    d512: T,
    d513: T,
    d514: T,
    d515: T,
    d516: T,

    d61: T, 
    d66: T, 
    d67: T, 
    d68: T, 
    d69: T, 
    d610: T,
    d611: T,
    d612: T,
    d613: T,
    d614: T,
    d615: T,
    d616: T,

    d71: T, 
    d76: T, 
    d77: T, 
    d78: T, 
    d79: T, 
    d710: T,
    d711: T,
    d712: T,
    d713: T,
    d714: T,
    d715: T,
    d716: T,
}

impl<T: Real, const R: usize, const C: usize, E: EventData> Solver<T, R, C, E> for DOP853<T, R, C, E> {    
    fn init<S>(&mut self, system: &S, t0: T, tf: T, y0: &SMatrix<T, R, C>)  -> Result<(), SolverStatus<T, R, C, E>>
    where 
        S: System<T, R, C, E>
    {
        // Set tf so step size doesn't go past it
        self.tf = tf;

        // Initialize Statistics
        self.evals = 0;
        self.steps = 0;
        self.rejected_steps = 0;
        self.accepted_steps = 0;

        // Set Current State as Initial State
        self.t = t0;
        self.y = y0.clone();

        // Calculate derivative at t0
        system.diff(t0, y0, &mut self.k1);
        self.evals += 1;

        // Initialize Previous State
        self.t_old = self.t;
        self.y_old = self.y;
        self.k_old = self.k1;

        // Calculate Initial Step
        if self.h0 == T::zero() {
            self.h_init(system, t0, tf);

            // Adjust h0 to be within bounds
            let posneg = (tf - t0).signum();
            if self.h0.abs() < self.h_min.abs() {
                self.h0 = self.h_min.abs() * posneg;
            } else if self.h0.abs() > self.h_max.abs() {
                self.h0 = self.h_max.abs() * posneg;
            }
        }

        // Check if h0 is within bounds, and h_min and h_max are valid
        match validate_step_size_parameters(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Set h_max to prevent single step overshoot
        self.h_max = match self.h_max {
            x if x > (self.tf - t0).abs() => (self.tf - t0).abs(),
            _ => self.h_max.abs(),
        };

        // Make sure iteration variables are reset
        self.h_lamb = T::zero();
        self.non_stiff_counter = 0;
        self.stiffness_counter = 0;

        // Solver is ready to go
        self.status = SolverStatus::Initialized;

        Ok(())
    }

    fn step<S>(&mut self, system: &S) 
    where 
        S: System<T, R, C, E>
    {
        // Check if Max Steps Reached
        if self.steps >= self.max_steps {
            self.status = SolverStatus::MaxSteps(self.t, self.y.clone());
            return;
        }

        // Check if Step Size is too smaller then machine default_epsilon
        if self.h.abs() < T::default_epsilon() {
            self.status = SolverStatus::StepSize(self.t, self.y.clone());
            return;
        }

        // The twelve stages
        system.diff(
            self.t + self.c2 * self.h,
            &(self.y + self.k1 * (self.a21 * self.h)),
            &mut self.k2
        );
        system.diff(
            self.t + self.c3 * self.h,
            &(self.y + self.k1 * (self.a31 * self.h) + self.k2 * (self.a32 * self.h)),
            &mut self.k3
        );
        system.diff(
            self.t + self.c4 * self.h,
            &(self.y + self.k1 * (self.a41 * self.h) + self.k3 * (self.a43 * self.h)),
            &mut self.k4
        );
        system.diff(
            self.t + self.c5 * self.h,
            &(self.y + self.k1 * (self.a51 * self.h) + self.k3 * (self.a53 * self.h) + self.k4 * (self.a54 * self.h)),
            &mut self.k5
        );
        system.diff(
            self.t + self.c6 * self.h,
            &(self.y + self.k1 * (self.a61 * self.h) + self.k4 * (self.a64 * self.h) + self.k5 * (self.a65 * self.h)),
            &mut self.k6
        );
        system.diff(
            self.t + self.c7 * self.h,
            &(self.y + self.k1 * (self.a71 * self.h) + self.k4 * (self.a74 * self.h) + self.k5 * (self.a75 * self.h) + self.k6 * (self.a76 * self.h)),
            &mut self.k7
        );
        system.diff(
            self.t + self.c8 * self.h,
            &(self.y + self.k1 * (self.a81 * self.h) + self.k4 * (self.a84 * self.h) + self.k5 * (self.a85 * self.h) + self.k6 * (self.a86 * self.h) + self.k7 * (self.a87 * self.h)),
            &mut self.k8
        );
        system.diff(
            self.t + self.c9 * self.h,
            &(self.y + self.k1 * (self.a91 * self.h) + self.k4 * (self.a94 * self.h) + self.k5 * (self.a95 * self.h) + self.k6 * (self.a96 * self.h) + self.k7 * (self.a97 * self.h) + self.k8 * (self.a98 * self.h)),
            &mut self.k9
        );
        system.diff(
            self.t + self.c10 * self.h,
            &(self.y + self.k1 * (self.a101 * self.h) + self.k4 * (self.a104 * self.h) + self.k5 * (self.a105 * self.h) + self.k6 * (self.a106 * self.h) + self.k7 * (self.a107 * self.h) + self.k8 * (self.a108 * self.h) + self.k9 * (self.a109 * self.h)),
            &mut self.k10
        );
        system.diff(
            self.t + self.c11 * self.h,
            &(self.y + self.k1 * (self.a111 * self.h) + self.k4 * (self.a114 * self.h) + self.k5 * (self.a115 * self.h) + self.k6 * (self.a116 * self.h) + self.k7 * (self.a117 * self.h) + self.k8 * (self.a118 * self.h) + self.k9 * (self.a119 * self.h) + self.k10 * (self.a1110 * self.h)),
            &mut self.k2
        );
        let t_new = self.t + self.h;
        let yy1 = self.y + self.k1 * (self.a121 * self.h) + self.k4 * (self.a124 * self.h) + self.k5 * (self.a125 * self.h) + self.k6 * (self.a126 * self.h) + self.k7 * (self.a127 * self.h) + self.k8 * (self.a128 * self.h) + self.k9 * (self.a129 * self.h) + self.k10 * (self.a1210 * self.h) + self.k2 * (self.a1211 * self.h);
        system.diff(
            t_new,
            &yy1,
            &mut self.k3
        );
        self.k4 = self.k1 * self.b1 + self.k6 * self.b6 + self.k7 * self.b7 + self.k8 * self.b8 + self.k9 * self.b9 + self.k10 * self.b10 + self.k2 * self.b11 + self.k3 * self.b12;
        self.k5 = self.y + self.k4 * self.h;
        
        self.evals += 11;

        // Error Estimation
        let mut err = T::zero();
        let mut err2 = T::zero();

        let n = self.y.len();
        for i in 0..n {
            let sk = self.atol + self.rtol * self.y[i].abs().max(self.k5[i].abs());
            let erri = self.k4[i] - self.bhh1 * self.k1[i] - self.bhh2 * self.k9[i] - self.bhh3 * self.k3[i];
            err2 = err2 + (erri / sk).powi(2);
            let erri = self.er1 * self.k1[i]
                + self.er6 * self.k6[i]
                + self.er7 * self.k7[i]
                + self.er8 * self.k8[i]
                + self.er9 * self.k9[i]
                + self.er10 * self.k10[i]
                + self.er11 * self.k2[i]
                + self.er12 * self.k3[i];
            err = err + (erri / sk).powi(2);
        }
        let mut deno = err + T::from_f64(0.01).unwrap() * err2;
        if deno <= T::zero() {
            deno = T::one();
        }
        err = self.h.abs() * err * (T::one() / (deno * T::from_usize(n).unwrap())).sqrt();

        // Computation of h_new
        self.fac11 = err.powf(self.expo1);
        // Lund-stabilization
        self.fac = self.fac11 / self.facold.powf(self.beta);
        // Requirement that fac1 <= h_new/h <= fac2
        self.fac = self.facc2.max(self.facc1.min(self.fac / self.safe));
        let mut h_new = self.h / self.fac;

        if err <= T::one() {
            // Step Accepted
            self.facold = err.max(T::from_f64(1.0e-4).unwrap());
            self.accepted_steps += 1;
            system.diff(t_new, &self.k5, &mut self.k4);
            self.evals += 1;

            // stiffness detection
            if self.accepted_steps % self.n_stiff == 0 {
                let mut stdnum = T::zero();
                let mut stden = T::zero();
                let sqr = self.k4 - self.k3;
                stdnum += sqr.component_mul(&sqr).sum();
                let sqr = self.k5 - yy1;
                stden += sqr.component_mul(&sqr).sum();

                if stden > T::zero() {
                    self.h_lamb = self.h * (stdnum / stden).sqrt();
                }
                if self.h_lamb > T::from_f64(6.1).unwrap() {
                    self.non_stiff_counter = 0;
                    self.stiffness_counter += 1;
                    if self.stiffness_counter == 15 {
                        // Early Exit Stiffness Detected
                        self.status = SolverStatus::Stiffness(self.t, self.y.clone());
                        return;
                    }
                } else {
                    self.non_stiff_counter += 1;
                    if self.non_stiff_counter == 6 {
                        self.stiffness_counter = 0;
                    }
                }
            }

            // Compute coefficients for interpolation


            // Conclusion of Step

            // For Interpolation
            self.y_old = self.y;
            self.k_old = self.k1;
            self.t_old = self.t;
            self.h_old = self.h;

            // Update State
            self.k1 = self.k4;
            self.y = self.k5;
            self.t = t_new;

            // Check if previous step rejected
            if let SolverStatus::RejectedStep = self.status {
                h_new = self.h.min(h_new);
                self.status = SolverStatus::Solving;
            }
        } else {
            // Step Rejected
            h_new = self.h / self.facc1.min(self.fac11 / self.safe);
            self.status = SolverStatus::RejectedStep;
            self.rejected_steps += 1;
        }
        // Step Complete
        self.h = constrain_step_size(h_new, self.h_min, self.h_max);

        self.steps += 1;
    }

    fn interpolate<F>(&mut self, system: &F, t: T) -> SMatrix<T, R, C>
    where
        F: System<T, R, C, E>
    {
        if self.cached_step_num != self.steps {
            // Initial coefficients
            self.cont1 = self.y_old;
            let ydiff = self.k5 - self.y_old;
            self.cont2 = ydiff;
            let bspl = self.k_old * self.h_old - ydiff;
            self.cont3 = bspl;
            self.cont4 = ydiff - self.k4 * self.h_old - bspl;
            self.cont5 = self.k_old * self.d41 + self.k6 * self.d46 + self.k7 * self.d47 + self.k8 * self.d48 +
                self.k9 * self.d49 + self.k10 * self.d410 + self.k2 * self.d411 + self.k3 * self.d412;
            self.cont6 = self.k_old * self.d51 + self.k6 * self.d56 + self.k7 * self.d57 + self.k8 * self.d58 +
                self.k9 * self.d59 + self.k10 * self.d510 + self.k2 * self.d511 + self.k3 * self.d512;
            self.cont7 = self.k_old * self.d61 + self.k6 * self.d66 + self.k7 * self.d67 + self.k8 * self.d68 +
                self.k9 * self.d69 + self.k10 * self.d610 + self.k2 * self.d611 + self.k3 * self.d612;
            self.cont8 = self.k_old * self.d71 + self.k6 * self.d76 + self.k7 * self.d77 + self.k8 * self.d78 +
                self.k9 * self.d79 + self.k10 * self.d710 + self.k2 * self.d711 + self.k3 * self.d712;

            // Next 3 Function Evaluations
            system.diff(
                self.t_old + self.c14 * self.h_old,
                &(self.y_old + (
                    self.k_old * self.a141 + self.k7 * self.a147 + self.k8 * self.a148 + self.k9 * self.a149 + self.k10 * self.a1410 + self.k2 * self.a1411 + self.k3 * self.a1412
                    + self.k4 * self.a1413) * self.h_old
                ),
                &mut self.k10
            );
            system.diff(
                self.t_old + self.c15 * self.h_old,
                &(self.y_old + (
                    self.k_old * self.a151 + self.k6 * self.a156 + self.k7 * self.a157 + self.k8 * self.a158 + self.k2 * self.a1511 + self.k3 * self.a1512 + self.k4 * self.a1513 + self.k10 * self.a1514) * self.h_old
                ),
                &mut self.k2
            );
            system.diff(
                self.t_old + self.c16 * self.h_old,
                &(self.y_old + (
                    self.k_old * self.a161 + self.k6 * self.a166 + self.k7 * self.a167 + self.k8 * self.a168 + self.k9 * self.a169 + self.k4 * self.a1613 + self.k10 * self.a1614 + self.k2 * self.a1615) * self.h_old
                ),
                &mut self.k3
            );
            self.evals += 3;

            // Final preparation
            self.cont5 = (self.cont5 + self.k4 * self.d413 + self.k10 * self.d414 + self.k2 * self.d415 + self.k3 * self.d416) * self.h_old;
            self.cont6 = (self.cont6 + self.k4 * self.d513 + self.k10 * self.d514 + self.k2 * self.d515 + self.k3 * self.d516) * self.h_old;
            self.cont7 = (self.cont7 + self.k4 * self.d613 + self.k10 * self.d614 + self.k2 * self.d615 + self.k3 * self.d616) * self.h_old;
            self.cont8 = (self.cont8 + self.k4 * self.d713 + self.k10 * self.d714 + self.k2 * self.d715 + self.k3 * self.d716) * self.h_old;

            // Step is cached
            self.cached_step_num = self.steps;
        }

        /* Interpolate for each desired step */
        let s = (t - self.t_old) / self.h_old;
        let s1 = T::one() - s;

        let conpar = self.cont5 + (self.cont6 + (self.cont7 + self.cont8 * s) * s1) * s;
        let y = self.cont1 + (self.cont2 + (self.cont3 + (self.cont4 + conpar * s1) * s) * s1) * s;

        y
    }

    fn t(&self) -> T {
        self.t
    }

    fn y(&self) -> &SMatrix<T, R, C> {
        &self.y
    }

    fn dydt(&self) -> &SMatrix<T, R, C> {
        &self.k1
    }

    fn t_prev(&self) -> T {
        self.t_old
    }

    fn y_prev(&self) -> &SMatrix<T, R, C> {
        &self.y_old
    }

    fn dydt_prev(&self) -> &SMatrix<T, R, C> {
        &self.k_old
    }

    fn h(&self) -> T {
        self.h
    }

    fn set_h(&mut self, h: T) {
        self.h = h;
    }

    fn evals(&self) -> usize {
        self.evals
    }

    fn steps(&self) -> usize {
        self.steps
    }

    fn rejected_steps(&self) -> usize {
        self.rejected_steps
    }

    fn accepted_steps(&self) -> usize {
        self.steps
    }

    fn status(&self) -> &SolverStatus<T, R, C, E> {
        &self.status
    }

    fn set_status(&mut self, status: SolverStatus<T, R, C, E>) {
        self.status = status;
    }
}

impl<T: Real, const R: usize, const C: usize, E: EventData> DOP853<T, R, C, E> {
    /// Creates a new DOP853 Solver.
    /// 
    /// # Returns
    /// * `system` - Function that defines the ordinary differential equation dy/dt = f(t, y).
    /// # Returns
    /// * DOP853 Struct ready to go for solving.
    ///  
    pub fn new() -> Self {
        DOP853 {
            ..Default::default()
        }
    }

    /// Initializes the initial step size for the solver.
    /// The initial step size is computed such that h**8 * f0.norm().max(der2.norm()) = 0.01
    /// 
    /// This function is called internally by the init function if non initial step size, h, is not provided.
    /// This function also dependents on derived settings and the initial derivative vector.
    /// Thus it is private and should not be called directly by users.
    /// 
    /// # Arguments
    /// * `ode` - Function that defines the ordinary differential equation dy/dt = f(t, y).
    /// 
    /// # Returns
    /// * Updates self.h with the initial step size.
    /// 
    fn h_init<S>(&mut self, system: &S, t0: T, tf: T)
    where 
        S: System<T, R, C, E>
    {
        // Set the initial step size h0 to h, if its 0.0 then it will be calculated
        self.h = self.h0;

        let posneg = (tf - t0).signum();

        let sk = (self.y.abs() * self.rtol).add_scalar(self.atol);
        let sqr = self.k1.component_div(&sk);
        let dnf = sqr.component_mul(&sqr).sum();
        let sqr = self.y.component_div(&sk);
        let dny = sqr.component_mul(&sqr).sum();

        self.h = if (dnf <= T::from_f64(1.0e-10).unwrap()) || (dny <= T::from_f64(1.0e-10).unwrap()) {
            T::from_f64(1.0e-6).unwrap()
        } else {
            (dny / dnf).sqrt() * T::from_f64(0.01).unwrap()
        };

        self.h = self.h.min(self.h_max);
        self.h = if posneg < T::zero() {
            -self.h.abs()
        } else {
            self.h.abs()
        };

        // perform an explicit Euler step
        system.diff(self.t + self.h, &(self.y + (self.k1 * self.h)), &mut self.k2);
        self.evals += 1;

        // estimate the second derivative of the solution
        let sk = (self.y.abs() * self.rtol).add_scalar(self.atol);
        let sqr = (self.k2 - self.k1).component_div(&sk);
        let der2 = (sqr.component_mul(&sqr)).sum().sqrt() / self.h;

        // step size is computed such that h**8 * f0.norm().max(der2.norm()) = 0.01
        let der12 = der2.abs().max(dnf.sqrt());
        let h1 = if der12 <= T::from_f64(1.0e-15).unwrap() {
            (self.h.abs() * T::from_f64(1.0e-3).unwrap()).max(T::from_f64(1.0e-6).unwrap())
        } else {
            (T::from_f64(0.01).unwrap() / der12).powf(T::one() / T::from_f64(8.0).unwrap())
        };

        self.h = (T::from_f64(100.0).unwrap() * posneg * self.h).min(h1.min(self.h_max));

        // Make sure step is going in the right direction
        self.h = self.h.abs() * posneg;
        self.h0 = self.h;
    }

    // Builder Functions
    pub fn rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    pub fn atol(mut self, atol: T) -> Self {
        self.atol = atol;
        self
    }

    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }

    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn n_stiff(mut self, n_stiff: usize) -> Self {
        self.n_stiff = n_stiff;
        self
    }

    pub fn safe(mut self, safe: T) -> Self {
        self.safe = safe;
        self
    }

    pub fn beta(mut self, beta: T) -> Self {
        self.beta = beta;
        self
    }

    pub fn fac1(mut self, fac1: T) -> Self {
        self.fac1 = fac1;
        self
    }

    pub fn fac2(mut self, fac2: T) -> Self {
        self.fac2 = fac2;
        self
    }

    pub fn expo1(mut self, expo1: T) -> Self {
        self.expo1 = expo1;
        self
    }

    pub fn facc1(mut self, facc1: T) -> Self {
        self.facc1 = facc1;
        self
    }

    pub fn facc2(mut self, facc2: T) -> Self {
        self.facc2 = facc2;
        self
    }
}

// Default Settings
impl<T: Real, const R: usize, const C: usize, E: EventData> Default for DOP853<T, R, C, E> {
    fn default() -> Self {
        DOP853 {
            t: T::zero(),
            y: SMatrix::zeros(),
            h: T::zero(),
            tf: T::zero(),
            h0: T::zero(),
            rtol: T::from_f64(1e-3).unwrap(),
            atol: T::from_f64(1e-6).unwrap(),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 1_000_000,
            n_stiff: 100,
            safe: T::from_f64(0.9).unwrap(),
            fac1: T::from_f64(0.33).unwrap(),
            fac2: T::from_f64(6.0).unwrap(),
            beta: T::from_f64(0.0).unwrap(),
            expo1: T::from_f64(1.0 / 8.0).unwrap(), //- beta / 0.0, beta = 0.0 so left out.
            facc1: T::from_f64(1.0 / 0.33).unwrap(),
            facc2: T::from_f64(1.0 / 6.0).unwrap(),
            facold: T::from_f64(1.0e-4).unwrap(),
            fac11: T::zero(),
            fac: T::zero(),
            status: SolverStatus::Uninitialized,
            h_lamb: T::zero(),
            non_stiff_counter: 0,
            stiffness_counter: 0,
            evals: 0,
            steps: 0,
            rejected_steps: 0,
            accepted_steps: 0,
            k1: SMatrix::zeros(),
            k2: SMatrix::zeros(),
            k3: SMatrix::zeros(),
            k4: SMatrix::zeros(),
            k5: SMatrix::zeros(),
            k6: SMatrix::zeros(),
            k7: SMatrix::zeros(),
            k8: SMatrix::zeros(),
            k9: SMatrix::zeros(),
            k10: SMatrix::zeros(),
            cached_step_num: 0,
            y_old: SMatrix::zeros(),
            k_old: SMatrix::zeros(),
            t_old: T::zero(),
            h_old: T::zero(),
            cont1: SMatrix::zeros(),
            cont2: SMatrix::zeros(),
            cont3: SMatrix::zeros(),
            cont4: SMatrix::zeros(),
            cont5: SMatrix::zeros(),
            cont6: SMatrix::zeros(),
            cont7: SMatrix::zeros(),
            cont8: SMatrix::zeros(),

            // Butcher Tableau Constants
            a21: T::from_f64(A21).unwrap(),
            a31: T::from_f64(A31).unwrap(),
            a32: T::from_f64(A32).unwrap(),
            a41: T::from_f64(A41).unwrap(),
            a43: T::from_f64(A43).unwrap(),
            a51: T::from_f64(A51).unwrap(),
            a53: T::from_f64(A53).unwrap(),
            a54: T::from_f64(A54).unwrap(),
            a61: T::from_f64(A61).unwrap(),
            a64: T::from_f64(A64).unwrap(),
            a65: T::from_f64(A65).unwrap(),
            a71: T::from_f64(A71).unwrap(),
            a74: T::from_f64(A74).unwrap(),
            a75: T::from_f64(A75).unwrap(),
            a76: T::from_f64(A76).unwrap(),

            a81: T::from_f64(A81).unwrap(),
            a84: T::from_f64(A84).unwrap(),
            a85: T::from_f64(A85).unwrap(),
            a86: T::from_f64(A86).unwrap(),
            a87: T::from_f64(A87).unwrap(),
            a91: T::from_f64(A91).unwrap(),
            a94: T::from_f64(A94).unwrap(),
            a95: T::from_f64(A95).unwrap(),
            a96: T::from_f64(A96).unwrap(),
            a97: T::from_f64(A97).unwrap(),
            a98: T::from_f64(A98).unwrap(),
            a101: T::from_f64(A101).unwrap(),
            a104: T::from_f64(A104).unwrap(),
            a105: T::from_f64(A105).unwrap(),
            a106: T::from_f64(A106).unwrap(),
            a107: T::from_f64(A107).unwrap(),
            a108: T::from_f64(A108).unwrap(),
            a109: T::from_f64(A109).unwrap(),

            a111: T::from_f64(A111).unwrap(),
            a114: T::from_f64(A114).unwrap(),
            a115: T::from_f64(A115).unwrap(),
            a116: T::from_f64(A116).unwrap(),
            a117: T::from_f64(A117).unwrap(),
            a118: T::from_f64(A118).unwrap(),
            a119: T::from_f64(A119).unwrap(),
            a1110: T::from_f64(A1110).unwrap(),
            a121: T::from_f64(A121).unwrap(),
            a124: T::from_f64(A124).unwrap(),
            a125: T::from_f64(A125).unwrap(),
            a126: T::from_f64(A126).unwrap(),
            a127: T::from_f64(A127).unwrap(),
            a128: T::from_f64(A128).unwrap(),
            a129: T::from_f64(A129).unwrap(),
            a1210: T::from_f64(A1210).unwrap(),
            a1211: T::from_f64(A1211).unwrap(),

            a141: T::from_f64(A141).unwrap(),
            a147: T::from_f64(A147).unwrap(),
            a148: T::from_f64(A148).unwrap(),
            a149: T::from_f64(A149).unwrap(),
            a1410: T::from_f64(A1410).unwrap(),
            a1411: T::from_f64(A1411).unwrap(),
            a1412: T::from_f64(A1412).unwrap(),
            a1413: T::from_f64(A1413).unwrap(),

            a151: T::from_f64(A151).unwrap(),
            a156: T::from_f64(A156).unwrap(),
            a157: T::from_f64(A157).unwrap(),
            a158: T::from_f64(A158).unwrap(),
            a1511: T::from_f64(A1511).unwrap(),
            a1512: T::from_f64(A1512).unwrap(),
            a1513: T::from_f64(A1513).unwrap(),
            a1514: T::from_f64(A1514).unwrap(),
            a161: T::from_f64(A161).unwrap(),
            a166: T::from_f64(A166).unwrap(),
            a167: T::from_f64(A167).unwrap(),
            a168: T::from_f64(A168).unwrap(),
            a169: T::from_f64(A169).unwrap(),
            a1613: T::from_f64(A1613).unwrap(),
            a1614: T::from_f64(A1614).unwrap(),
            a1615: T::from_f64(A1615).unwrap(),

            b1: T::from_f64(B1).unwrap(),
            b6: T::from_f64(B6).unwrap(),
            b7: T::from_f64(B7).unwrap(),
            b8: T::from_f64(B8).unwrap(),
            b9: T::from_f64(B9).unwrap(),
            b10: T::from_f64(B10).unwrap(),
            b11: T::from_f64(B11).unwrap(),
            b12: T::from_f64(B12).unwrap(),

            bhh1: T::from_f64(BHH1).unwrap(),
            bhh2: T::from_f64(BHH2).unwrap(),
            bhh3: T::from_f64(BHH3).unwrap(),

            c2: T::from_f64(C2).unwrap(),
            c3: T::from_f64(C3).unwrap(),
            c4: T::from_f64(C4).unwrap(),
            c5: T::from_f64(C5).unwrap(),
            c6: T::from_f64(C6).unwrap(),
            c7: T::from_f64(C7).unwrap(),
            c8: T::from_f64(C8).unwrap(),
            c9: T::from_f64(C9).unwrap(),
            c10: T::from_f64(C10).unwrap(),
            c11: T::from_f64(C11).unwrap(),
            c14: T::from_f64(C14).unwrap(),
            c15: T::from_f64(C15).unwrap(),
            c16: T::from_f64(C16).unwrap(),

            er1: T::from_f64(ER1).unwrap(),
            er6: T::from_f64(ER6).unwrap(),
            er7: T::from_f64(ER7).unwrap(),
            er8: T::from_f64(ER8).unwrap(),
            er9: T::from_f64(ER9).unwrap(),
            er10: T::from_f64(ER10).unwrap(),
            er11: T::from_f64(ER11).unwrap(),
            er12: T::from_f64(ER12).unwrap(),

            d41: T::from_f64(D41).unwrap(),
            d46: T::from_f64(D46).unwrap(),
            d47: T::from_f64(D47).unwrap(),
            d48: T::from_f64(D48).unwrap(),
            d49: T::from_f64(D49).unwrap(),
            d410: T::from_f64(D410).unwrap(),
            d411: T::from_f64(D411).unwrap(),
            d412: T::from_f64(D412).unwrap(),
            d413: T::from_f64(D413).unwrap(),
            d414: T::from_f64(D414).unwrap(),
            d415: T::from_f64(D415).unwrap(),
            d416: T::from_f64(D416).unwrap(),
            d51: T::from_f64(D51).unwrap(),
            d56: T::from_f64(D56).unwrap(),
            d57: T::from_f64(D57).unwrap(),
            d58: T::from_f64(D58).unwrap(),
            d59: T::from_f64(D59).unwrap(),

            d510: T::from_f64(D510).unwrap(),
            d511: T::from_f64(D511).unwrap(),
            d512: T::from_f64(D512).unwrap(),
            d513: T::from_f64(D513).unwrap(),
            d514: T::from_f64(D514).unwrap(),
            d515: T::from_f64(D515).unwrap(),
            d516: T::from_f64(D516).unwrap(),

            d61: T::from_f64(D61).unwrap(),
            d66: T::from_f64(D66).unwrap(),
            d67: T::from_f64(D67).unwrap(),
            d68: T::from_f64(D68).unwrap(),
            d69: T::from_f64(D69).unwrap(),
            d610: T::from_f64(D610).unwrap(),
            d611: T::from_f64(D611).unwrap(),
            d612: T::from_f64(D612).unwrap(),
            d613: T::from_f64(D613).unwrap(),
            d614: T::from_f64(D614).unwrap(),
            d615: T::from_f64(D615).unwrap(),
            d616: T::from_f64(D616).unwrap(),

            d71: T::from_f64(D71).unwrap(),
            d76: T::from_f64(D76).unwrap(),
            d77: T::from_f64(D77).unwrap(),
            d78: T::from_f64(D78).unwrap(),
            d79: T::from_f64(D79).unwrap(),
            d710: T::from_f64(D710).unwrap(),
            d711: T::from_f64(D711).unwrap(),
            d712: T::from_f64(D712).unwrap(),
            d713: T::from_f64(D713).unwrap(),
            d714: T::from_f64(D714).unwrap(),
            d715: T::from_f64(D715).unwrap(),
            d716: T::from_f64(D716).unwrap(),
        }
    }
}

// Butcher Tableau for DOP853
const A21: f64 = 5.26001519587677318785587544488E-2;
const A31: f64 = 1.97250569845378994544595329183E-2;
const A32: f64 = 5.91751709536136983633785987549E-2;
const A41: f64 = 2.95875854768068491816892993775E-2;
const A43: f64 = 8.87627564304205475450678981324E-2;
const A51: f64 = 2.41365134159266685502369798665E-1;
const A53: f64 = -8.84549479328286085344864962717E-1;
const A54: f64 = 9.24834003261792003115737966543E-1;
const A61: f64 = 3.7037037037037037037037037037E-2;
const A64: f64 = 1.70828608729473871279604482173E-1;
const A65: f64 = 1.25467687566822425016691814123E-1;
const A71: f64 = 3.7109375E-2;
const A74: f64 = 1.70252211019544039314978060272E-1;
const A75: f64 = 6.02165389804559606850219397283E-2;
const A76: f64 = -1.7578125E-2;

const A81: f64 = 3.70920001185047927108779319836E-2;
const A84: f64 = 1.70383925712239993810214054705E-1;
const A85: f64 = 1.07262030446373284651809199168E-1;
const A86: f64 = -1.53194377486244017527936158236E-2;
const A87: f64 = 8.27378916381402288758473766002E-3;
const A91: f64 = 6.24110958716075717114429577812E-1;
const A94: f64 = -3.36089262944694129406857109825E0;
const A95: f64 = -8.68219346841726006818189891453E-1;
const A96: f64 = 2.75920996994467083049415600797E1;
const A97: f64 = 2.01540675504778934086186788979E1;
const A98: f64 = -4.34898841810699588477366255144E1;
const A101: f64 = 4.77662536438264365890433908527E-1;
const A104: f64 = -2.48811461997166764192642586468E0;
const A105: f64 = -5.90290826836842996371446475743E-1;
const A106: f64 = 2.12300514481811942347288949897E1;
const A107: f64 = 1.52792336328824235832596922938E1;
const A108: f64 = -3.32882109689848629194453265587E1;
const A109: f64 = -2.03312017085086261358222928593E-2;

const A111: f64 = -9.3714243008598732571704021658E-1;
const A114: f64 = 5.18637242884406370830023853209E0;
const A115: f64 = 1.09143734899672957818500254654E0;
const A116: f64 = -8.14978701074692612513997267357E0;
const A117: f64 = -1.85200656599969598641566180701E1;
const A118: f64 = 2.27394870993505042818970056734E1;
const A119: f64 = 2.49360555267965238987089396762E0;
const A1110: f64 = -3.0467644718982195003823669022E0;
const A121: f64 = 2.27331014751653820792359768449E0;
const A124: f64 = -1.05344954667372501984066689879E1;
const A125: f64 = -2.00087205822486249909675718444E0;
const A126: f64 = -1.79589318631187989172765950534E1;
const A127: f64 = 2.79488845294199600508499808837E1;
const A128: f64 = -2.85899827713502369474065508674E0;
const A129: f64 = -8.87285693353062954433549289258E0;
const A1210: f64 = 1.23605671757943030647266201528E1;
const A1211: f64 = 6.43392746015763530355970484046E-1;

const A141: f64 = 5.61675022830479523392909219681E-2;
const A147: f64 = 2.53500210216624811088794765333E-1;
const A148: f64 = -2.46239037470802489917441475441E-1;
const A149: f64 = -1.24191423263816360469010140626E-1;
const A1410: f64 = 1.5329179827876569731206322685E-1;
const A1411: f64 = 8.20105229563468988491666602057E-3;
const A1412: f64 = 7.56789766054569976138603589584E-3;
const A1413: f64 = -8.298E-3;

const A151: f64 = 3.18346481635021405060768473261E-2;
const A156: f64 = 2.83009096723667755288322961402E-2;
const A157: f64 = 5.35419883074385676223797384372E-2;
const A158: f64 = -5.49237485713909884646569340306E-2;
const A1511: f64 = -1.08347328697249322858509316994E-4;
const A1512: f64 = 3.82571090835658412954920192323E-4;
const A1513: f64 = -3.40465008687404560802977114492E-4;
const A1514: f64 = 1.41312443674632500278074618366E-1;
const A161: f64 = -4.28896301583791923408573538692E-1;
const A166: f64 = -4.69762141536116384314449447206E0;
const A167: f64 = 7.68342119606259904184240953878E0;
const A168: f64 = 4.06898981839711007970213554331E0;
const A169: f64 = 3.56727187455281109270669543021E-1;
const A1613: f64 = -1.39902416515901462129418009734E-3;
const A1614: f64 = 2.9475147891527723389556272149E0;
const A1615: f64 = -9.15095847217987001081870187138E0;

const B1: f64 = 5.42937341165687622380535766363E-2;
const B6: f64 = 4.45031289275240888144113950566E0;
const B7: f64 = 1.89151789931450038304281599044E0;
const B8: f64 = -5.8012039600105847814672114227E0;
const B9: f64 = 3.1116436695781989440891606237E-1;
const B10: f64 = -1.52160949662516078556178806805E-1;
const B11: f64 = 2.01365400804030348374776537501E-1;
const B12: f64 = 4.47106157277725905176885569043E-2;

const BHH1: f64 = 0.244094488188976377952755905512E+00;
const BHH2: f64 = 0.733846688281611857341361741547E+00;
const BHH3: f64 = 0.220588235294117647058823529412E-01;

const C2: f64 = 0.526001519587677318785587544488E-01;
const C3: f64 = 0.789002279381515978178381316732E-01;
const C4: f64 = 0.118350341907227396726757197510E+00;
const C5: f64 = 0.281649658092772603273242802490E+00;
const C6: f64 = 0.333333333333333333333333333333E+00;
const C7: f64 = 0.25E+00;
const C8: f64 = 0.307692307692307692307692307692E+00;
const C9: f64 = 0.651282051282051282051282051282E+00;
const C10: f64 = 0.6E+00;
const C11: f64 = 0.857142857142857142857142857142E+00;
const C14: f64 = 0.1E+00;
const C15: f64 = 0.2E+00;
const C16: f64 = 0.777777777777777777777777777778E+00;

const ER1: f64 = 0.1312004499419488073250102996E-01;
const ER6: f64 = -0.1225156446376204440720569753E+01;
const ER7: f64 = -0.4957589496572501915214079952E+00;
const ER8: f64 = 0.1664377182454986536961530415E+01;
const ER9: f64 = -0.3503288487499736816886487290E+00;
const ER10: f64 = 0.3341791187130174790297318841E+00;
const ER11: f64 = 0.8192320648511571246570742613E-01;
const ER12: f64 = -0.2235530786388629525884427845E-01;

const D41: f64 = -0.84289382761090128651353491142E+01;
const D46: f64 = 0.56671495351937776962531783590E+00;
const D47: f64 = -0.30689499459498916912797304727E+01;
const D48: f64 = 0.23846676565120698287728149680E+01;
const D49: f64 = 0.21170345824450282767155149946E+01;
const D410: f64 = -0.87139158377797299206789907490E+00;
const D411: f64 = 0.22404374302607882758541771650E+01;
const D412: f64 = 0.63157877876946881815570249290E+00;
const D413: f64 = -0.88990336451333310820698117400E-01;
const D414: f64 = 0.18148505520854727256656404962E+02;
const D415: f64 = -0.91946323924783554000451984436E+01;
const D416: f64 = -0.44360363875948939664310572000E+01;

const D51: f64 = 0.10427508642579134603413151009E+02;
const D56: f64 = 0.24228349177525818288430175319E+03;
const D57: f64 = 0.16520045171727028198505394887E+03;
const D58: f64 = -0.37454675472269020279518312152E+03;
const D59: f64 = -0.22113666853125306036270938578E+02;
const D510: f64 = 0.77334326684722638389603898808E+01;
const D511: f64 = -0.30674084731089398182061213626E+02;
const D512: f64 = -0.93321305264302278729567221706E+01;
const D513: f64 = 0.15697238121770843886131091075E+02;
const D514: f64 = -0.31139403219565177677282850411E+02;
const D515: f64 = -0.93529243588444783865713862664E+01;
const D516: f64 = 0.35816841486394083752465898540E+02;

const D61: f64 = 0.19985053242002433820987653617E+02;
const D66: f64 = -0.38703730874935176555105901742E+03;
const D67: f64 = -0.18917813819516756882830838328E+03;
const D68: f64 = 0.52780815920542364900561016686E+03;
const D69: f64 = -0.11573902539959630126141871134E+02;
const D610: f64 = 0.68812326946963000169666922661E+01;
const D611: f64 = -0.10006050966910838403183860980E+01;
const D612: f64 = 0.77771377980534432092869265740E+00;
const D613: f64 = -0.27782057523535084065932004339E+01;
const D614: f64 = -0.60196695231264120758267380846E+02;
const D615: f64 = 0.84320405506677161018159903784E+02;
const D616: f64 = 0.11992291136182789328035130030E+02;

const D71: f64 = -0.25693933462703749003312586129E+02;
const D76: f64 = -0.15418974869023643374053993627E+03;
const D77: f64 = -0.23152937917604549567536039109E+03;
const D78: f64 = 0.35763911791061412378285349910E+03;
const D79: f64 = 0.93405324183624310003907691704E+02;
const D710: f64 = -0.37458323136451633156875139351E+02;
const D711: f64 = 0.10409964950896230045147246184E+03;
const D712: f64 = 0.29840293426660503123344363579E+02;
const D713: f64 = -0.43533456590011143754432175058E+02;
const D714: f64 = 0.96324553959188282948394950600E+02;
const D715: f64 = -0.39177261675615439165231486172E+02;
const D716: f64 = -0.14972683625798562581422125276E+03;