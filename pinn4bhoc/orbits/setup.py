import numpy as np

# domain of PINN
#          phi,   u0,   v0
LOWER = [ 0.00, 0.10, -1.0]
UPPER = [ DPHI, 0.99,  1.0]

# initial conditions for boomerang orbits
INIT_COND = np.array([
#r          delta
[1.01, 165.139495],
[1.20, 117.820657],
[1.40,  97.250387],
[1.50,  90.000000],
[2.00,  66.827523],
[3.00,  45.236645],
[5.00,  27.993690]
])
# colors of boomerang orbits
COLOR = ['blue', 'steelblue', 'green', 
         'goldenrod', 'darkgoldenrod', 
         'brown', 'red']
# ----------------------------------                
# convert to u0, v0
# ----------------------------------
R0 = INIT_COND[:, 0]

# photon emission angles
DELTA = INIT_COND[:, 1] * np.pi / 180

# initial conditions in (u, v) space
U0 = 1/R0
V0 = U0 * np.sqrt(1 - U0) / np.tan(DELTA)
