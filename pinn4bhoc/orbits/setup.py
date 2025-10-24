import numpy as np

# Boomerang orbit parameters
# ---------------------------------------
# domain of PINN
DPHI = 0.1  # phi slice
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
def compute_u0_v0(r0, delta, in_degrees=True, eps=np.pi/1000):
    assert(r0 > 1)
    u0 = 1 / r0
    
    if in_degrees:
        angle = delta * np.pi / 180
    else:
        angle = delta
    assert(np.pi-eps > delta > eps)

    v0 = u0 * np.sqrt(1-u0) / np.tan(delta)
    return  u0, v0

# photon initial radial coordinates in units of the Schwarzschild radius
R0 = INIT_COND[:, 0]

# photon emission angles (in degrees)
DELTA = INIT_COND[:, 1]

# initial conditions in (u, v) space
U0, V0 = compute_u0_v0(R0, DELTA)

