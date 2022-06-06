Project name: "Efficient MPC control of the robot leg"
Purpose create fast MPC controller for robot leg model.
Casadi test - is file with final fast casadi mpc

So just run Casadi_test (I descrybed some functions forward and you need to run next cells if some names are doesnt defined)
Fit whatever you want in MPC cell. T in this file is time MPC horizon. N in this file is MPC prediction steps, time for every step is T/N.
Other files are just drafts for this file (I calculated something and put into Casadi_test for example matrises).

Powerpoint contains main approaches, goals, steps.

homogeniuse_coord -  is file where i described with sympy Math model for leg.
Matrix_for_py is file with fist approach i mean fist MPC try
There are also presentation.
