import numpy as np
from scipy.integrate import odeint

# This code implements the 2-loop RGE running for gauge couplings and 1-loop for Yukawa couplings in the MSSM.
# For Yukawa, the 2-loop is complicated and omitted for simplicity; it's negligible for lighter generations.
# We assume diagonal Yukawa couplings with no mixing, running each diagonal element as a scalar.
# Traces are approximated using third generation dominance for light Yukawa running.
# To run from low energy (MZ) to high energy (GUT), we use t = ln(mu / MZ), with negative t for high to low, but here we run upward.
# For the paper, to match the calculation, run from GUT to SUSY scale, then add threshold corrections.

# Constants
MZ = 91.1876  # GeV
MGUT = 2e16  # Approximate GUT scale in GeV

# Initial values at MZ (approximate values; adjust as needed)
alpha1_MZ = 0.016887  # alpha1 = g1^2 / (4 pi), with g1 = sqrt(5/3) g'
alpha2_MZ = 0.03378
alpha3_MZ = 0.1184

g1_MZ = np.sqrt(4 * np.pi * alpha1_MZ)
g2_MZ = np.sqrt(4 * np.pi * alpha2_MZ)
g3_MZ = np.sqrt(4 * np.pi * alpha3_MZ)

# Third generation Yukawa at MZ (approximate, tan beta dependent; assume tan beta = 10)
yt_MZ = 0.99  # top
yb_MZ = 0.016
ytau_MZ = 0.01

# Light Yukawa at MZ for example; but for unification, we will run from GUT
# For demonstration, we set initial at MZ, but for paper, reverse the running.

def beta_func(y, t):
    # y = [g1, g2, g3, yt, yb, ytau, ys, yd, ymu, ye, yb (redundant, but for full)]
    # For light, ys, yd, ymu, ye
    g1, g2, g3, yt, yb, ytau, ys, yd, ymu, ye = y

    # Traces for Yukawa (approximate, assuming diagonal and third dominant)
    tr Yu2 = yt**2  # 3 generations, but light small
    tr Yd2 = yb**2 + ys**2 + yd**2
    tr Ye2 = ytau**2 + ymu**2 + ye**2

    # But for light, they are small, so tr Yu2 ≈ yt**2, etc.

    # Gauge RGE at 2-loop
    b1 = 33.0 / 5
    b2 = 1.0
    b3 = 3.0

    B = np.array([
        [199.0/25, 27.0/5, 88.0/5],
        [9.0/5, 25.0, 24.0],
        [11.0/5, 9.0, 14.0]
    ])

    # Yukawa contributions to gauge 2-loop (negative contribution to beta(g))
    y contrib1 = (17.0/5) * tr Yu2 + (3.0/5) * tr Yd2 + (3.0/5) * tr Ye2
    y contrib2 = 3 * tr Yu2 + 3 * tr Yd2 + tr Ye2
    y contrib3 = 2 * tr Yu2 + 2 * tr Yd2

    # beta(g_i) = g_i^3 / (16 pi^2) * b_i + g_i^3 / (16 pi^2)^2 * sum_j B_ij g_j^2 - g_i^3 / (16 pi^2)^2 * y_contrib_i
    beta_g1 = g1**3 / (16 * np.pi**2 ) * b1 + g1**3 / (16 * np.pi**2 )**2 * ( B[0,0] * g1**2 + B[0,1] * g2**2 + B[0,2] * g3**2 ) - g1**3 / (16 * np.pi**2 )**2 * y contrib1

    beta_g2 = g2**3 / (16 * np.pi**2 ) * b2 + g2**3 / (16 * np.pi**2 )**2 * ( B[1,0] * g1**2 + B[1,1] * g2**2 + B[1,1] * g3**2 ) - g2**3 / (16 * np.pi**2 )**2 * y contrib2

    beta_g3 = g3**3 / (16 * np.pi**2 ) * b3 + g3**3 / (16 * np.pi**2 )**2 * ( B[2,0] * g1**2 + B[2,1] * g2**2 + B[2,2] * g3**2 ) - g3**3 / (16 * np.pi**2 )**2 * y_contrib3

    # Yukawa RGE at 1-loop (for 2-loop, add more terms; see Martin & Vaughn for full)
    # For yt (up type)
    beta_yt = (1 / (16 * np.pi**2 )) * yt * ( 6 * yt**2 + yb**2 + 3 * tr Yu2 + tr Yd2 - (16/3 * g3**2 + 3 * g2**2 + 13/15 * g1**2) )

    # For yb (down type)
    beta_yb = (1 / (16 * np.pi**2 )) * yb * ( 6 * yb**2 + yt**2 + tr Ye2 + 3 * tr Yd2 - (16/3 * g3**2 + 3 * g2**2 + 1/15 * g1**2) )

    # For ytau (lepton)
    beta_ytau = (1 / (16 * np.pi**2 )) * ytau * ( 4 * ytau**2 + tr Ye2 + 3 * tr Yd2 - (3 * g2**2 + 9/5 * g1**2) )

    # For ys (strange, down type, small, neglect self y^3)
    beta_ys = (1 / (16 * np.pi**2 )) * ys * ( tr Ye2 + 3 * tr Yd2 - (16/3 * g3**2 + 3 * g2**2 + 1/15 * g1**2) )

    # For yd (down, same as ys)
    beta_yd = beta_ys.replace(ys, yd)  # Same formula

    # For ymu (muon, lepton)
    beta_ymu = (1 / (16 * np.pi**2 )) * ymu * ( tr Ye2 + 3 * tr Yd2 - (3 * g2**2 + 9/5 * g1**2) )

    # For ye (electron, same as ymu)
    beta_ye = beta_ymu.replace(ymu, ye)

    return [beta_g1, beta_g2, beta_g3, beta_yt, beta_yb, beta_ytau, beta_ys, beta_yd, beta_ymu, beta_ye]

# Initial condition at MZ
y0 = [g1_MZ, g2_MZ, g3_MZ, yt_MZ, yb_MZ, ytau_MZ, 0.003, 0.0001, 0.06, 0.0005]  # approximate ys, yd, ymu, ye at MZ

# t range from MZ to GUT, t = ln(mu / MZ), so from 0 to ln(MGUT / MZ)
t = np.logspace(0, np.log(MGUT / MZ), 1000)

# Solve
sol = odeint(beta_func, y0, t)

# To run from GUT to low, use negative t, from 0 to -ln(MGUT / MZ), and reverse the beta sign, or integrate backward.
# For example, to run down, def beta_down(y, t): return -beta_func(y, -t)

# Print example
print("g1 at GUT:", sol[-1,0])
# Add threshold calculations as per paper, using the formulas in appendix.

# To add 2-loop Yukawa, implement the full expressions from hep-ph/9308222 appendix.
# For example, for beta2_yt = (1/(16 pi^2)^2) * ... long expression.
# The code can be extended accordingly.
