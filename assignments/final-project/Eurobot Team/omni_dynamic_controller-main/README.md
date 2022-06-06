# omni_dynamic_controller

This repository contains a ros package with MPC and LQR controller for controlling the GUIDO robot by RESET. 

## Plan 
To complete the task of synthesising the controllers we've came up with a series of goals:
1. Mathematical model synthesis
2. Model verification
3. Controller design: LQR & MPC
4. Implementation
5. Results evaluation

## Installation 

### Prerequests

1. **do_mpc** library. Download this by using 
```
pip install do_mpc
```
2. **eurobot** workspace, made by RESET team.

## Running
1. Clone this repository to the eurobot workspace.
2. Launch the desired controller node: 
for LQR use:
```
roslaunch omni_dynamic_controller lqr_controller.launch
```
for MPC use:
```
roslaunch omni_dynamic_controller mpc_controller.launch
```
