//! Adaptive step size Runge-Kutta methods

use crate::adaptive_runge_kutta_method;

adaptive_runge_kutta_method!(
    /// Runge-Kutta-Fehlberg 4(5) adaptive method
    /// This method uses six function evaluations to calculate a fifth-order accurate
    /// solution, with an embedded fourth-order method for error estimation.
    /// The RKF45 method is one of the most widely used adaptive step size methods due to
    /// its excellent balance of efficiency and accuracy.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0      |
    /// 1/4    | 1/4
    /// 3/8    | 3/32         9/32
    /// 12/13  | 1932/2197    -7200/2197  7296/2197
    /// 1      | 439/216      -8          3680/513    -845/4104
    /// 1/2    | -8/27        2           -3544/2565  1859/4104   -11/40
    /// -----------------------------------------------------------------------
    ///        | 16/135       0           6656/12825  28561/56430 -9/50       2/55    (5th order)
    ///        | 25/216       0           1408/2565   2197/4104   -1/5        0       (4th order)
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method#CITEREFFehlberg1969)
    name: RKF,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
        [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
        [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0, 0.0],
        [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0.0]
    ],
    b: [
        [16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0], // 5th order
        [25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0]           // 4th order
    ],
    c: [0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0],
    order: 5,
    stages: 6
);

adaptive_runge_kutta_method!(
    /// Cash-Karp 4(5) adaptive method
    /// This method uses six function evaluations to calculate a fifth-order accurate
    /// solution, with an embedded fourth-order method for error estimation.
    /// The Cash-Karp method is a variant of the Runge-Kutta-Fehlberg method that uses
    /// different coefficients to achieve a more efficient and accurate solution.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0      |
    /// 1/5    | 1/5
    /// 3/10   | 3/40         9/40
    /// 3/5    | 3/10         -9/10       6/5
    /// 1      | -11/54       5/2         -70/27      35/27
    /// 7/8    | 1631/55296   175/512     575/13824   44275/110592 253/4096
    /// ------------------------------------------------------------------------------------
    ///        | 37/378       0           250/621     125/594     0           512/1771  (5th order)
    ///        | 2825/27648   0           18575/48384 13525/55296 277/14336   1/4       (4th order)
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method)
    name: CashKarp,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/10.0, -9.0/10.0, 6.0/5.0, 0.0, 0.0, 0.0],
        [-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0, 0.0, 0.0],
        [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, 0.0]
    ],
    b: [
        [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0], // 5th order
        [2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0] // 4th order
    ],
    c: [0.0, 1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0],
    order: 5,
    stages: 6
);

adaptive_runge_kutta_method!(
    /// Dormand-Prince 5(4) adaptive method
    /// This method uses seven function evaluations to calculate a fifth-order accurate 
    /// solution, with an embedded fourth-order method for error estimation.
    /// The DOPRI5 method is one of the most widely used adaptive step size methods due to
    /// its excellent balance of efficiency and accuracy.
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0      |
    /// 1/5    | 1/5
    /// 3/10   | 3/40         9/40
    /// 4/5    | 44/45        -56/15      32/9
    /// 8/9    | 19372/6561   -25360/2187 64448/6561   -212/729
    /// 1      | 9017/3168    -355/33     46732/5247   49/176        -5103/18656
    /// 1      | 35/384       0           500/1113     125/192       -2187/6784    11/84
    /// ----------------------------------------------------------------------------------------------
    ///        | 35/384       0           500/1113     125/192       -2187/6784    11/84       0       (5th order)
    ///        | 5179/57600   0           7571/16695   393/640       -92097/339200 187/2100    1/40    (4th order)
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method)
    name: DOPRI5,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
    ],
    b: [
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0], // 5th order
        [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0] // 4th order
    ],
    c: [0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0],
    order: 5,
    stages: 7
);