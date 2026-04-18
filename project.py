#!/usr/bin/env python3
"""
=============================================================================
  Schottky Contact Simulation — DEVSIM  (Au / n-Si, 1D, 300 K)
=============================================================================
  Fixes applied vs. original project.py
  ──────────────────────────────────────
  1.  Proper multi-point mesh with fine spacing near Schottky contact
  2.  Correct add_1d_contact API  (tag=, not pos=/region=)
  3.  All solution variables declared  (Potential, Electrons, Holes)
  4.  Full physics registered:  Poisson + drift-diffusion (SG) + SRH
  5.  Explicit Bernoulli function  (SYMDIFF has no bernoulli() built-in)
  6.  Proper Schottky BC  (thermionic emission, Crowell & Sze 1966)
  7.  Proper ohmic BC  (charge-neutral equilibrium)
  8.  Warm-start initial conditions  (bulk quasi-neutral guess)
  9.  Equilibrium solve before bias sweep
  10. Voltage ramp  (small steps, reverse + forward bias)
  11. Correct get_contact_current call  (equation= required)
  12. Auto math-library resolver for Windows MKL/OpenBLAS
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  MATH LIBRARY RESOLVER  (Windows — must run before import devsim)
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, glob

def _resolve_math_libs():
    if "DEVSIM_MATH_LIBS" in os.environ:
        print(f"[math-resolver] Using existing DEVSIM_MATH_LIBS="
              f"{os.environ['DEVSIM_MATH_LIBS']}")
        return
    if sys.platform != "win32":
        return
    py_root   = os.path.dirname(sys.executable)
    site_pkgs = os.path.join(py_root, "Lib", "site-packages")
    candidates = [
        ("mkl_rt.2.dll",   site_pkgs),
        ("mkl_rt.1.dll",   site_pkgs),
        ("mkl_rt.dll",     site_pkgs),
        ("mkl_rt.2.dll",   py_root),
        ("mkl_rt.1.dll",   py_root),
        ("mkl_rt.dll",     py_root),
        ("libopenblas*.dll", site_pkgs),
    ]
    for pattern, root in candidates:
        hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
        if hits:
            os.environ["DEVSIM_MATH_LIBS"] = hits[0]
            print(f"[math-resolver] Found: {hits[0]}")
            return
    print("[math-resolver] WARNING: No BLAS/LAPACK DLL found.\n"
          "  Run:  pip install mkl  then retry.")

_resolve_math_libs()

import math
import csv
import numpy as np
import devsim

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL CONSTANTS  (CODATA 2018)
# ─────────────────────────────────────────────────────────────────────────────
q     = 1.602176634e-19   # Elementary charge [C]
k_B   = 1.380649e-23      # Boltzmann constant [J/K]
eps0  = 8.8541878128e-14  # Permittivity of free space [F/cm]
T     = 300.0             # Temperature [K]
V_T   = k_B * T / q      # Thermal voltage ≈ 0.02585 V

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SILICON MATERIAL PARAMETERS  (T = 300 K, Sze Table 2.4)
# ─────────────────────────────────────────────────────────────────────────────
eps_r  = 11.7
eps_Si = eps_r * eps0     # 1.036e-12 F/cm

N_C   = 2.8e19            # CB effective DOS [cm⁻³]
N_V   = 1.04e19           # VB effective DOS [cm⁻³]
Eg    = 1.12              # Bandgap [eV]
n_i   = math.sqrt(N_C * N_V * math.exp(-Eg / V_T))   # ≈ 9.65e9 cm⁻³

mu_n  = 1400.0            # Electron mobility [cm²/V·s]
mu_p  = 450.0             # Hole mobility [cm²/V·s]
D_n   = mu_n * V_T        # Einstein: D = μ kT/q
D_p   = mu_p * V_T

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DOPING  (n-type, N_D = 10¹⁶ cm⁻³)
# ─────────────────────────────────────────────────────────────────────────────
N_D   = 1.0e16
N_A   = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SCHOTTKY CONTACT  (Au / n-Si)
# ─────────────────────────────────────────────────────────────────────────────
phi_B  = 0.80             # Barrier height [eV]  (experimental, Sze Table 5.1)
A_star = 110.0            # Richardson constant [A/(cm²·K²)]
v_n    = A_star * T**2 / (q * N_C)          # Thermionic velocity [cm/s]
n_s0   = N_C * math.exp(-phi_B / V_T)       # Equilibrium interface density
p_s0   = n_i**2 / n_s0

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SRH RECOMBINATION
# ─────────────────────────────────────────────────────────────────────────────
tau_n  = 1.0e-6           # Electron lifetime [s]
tau_p  = 1.0e-6           # Hole lifetime [s]

# ─────────────────────────────────────────────────────────────────────────────
# 6.  OHMIC (BACK) CONTACT EQUILIBRIUM VALUES
# ─────────────────────────────────────────────────────────────────────────────
n_eq_back  = N_D
p_eq_back  = n_i**2 / N_D
phi_back   = V_T * math.log(N_D / n_i)     # ≈ 0.347 V above midgap

# ─────────────────────────────────────────────────────────────────────────────
# 7.  MESH
#     Fine region  0 → 300 nm:  1.5 nm spacing  (resolves depletion layer)
#     Coarse region 300 nm → 1 μm:  7 nm spacing  (bulk)
# ─────────────────────────────────────────────────────────────────────────────
MESH   = "mesh1"
DEV    = "D1"
REG    = "Si"
SCK    = "anode"
OHM    = "cathode"

devsim.create_1d_mesh(mesh=MESH)

x_fine_end = 3.0e-5      # 300 nm
dx_fine    = 1.5e-7      # 1.5 nm
dx_bulk    = 7.0e-7      # 7.0 nm
L          = 1.0e-4      # 1 μm

Nfine = int(x_fine_end / dx_fine)
for i in range(Nfine + 1):
    devsim.add_1d_mesh_line(mesh=MESH, pos=i * dx_fine,
                             ps=dx_fine, tag=f"f{i}")

Nbulk = int((L - x_fine_end) / dx_bulk)
for i in range(1, Nbulk + 1):
    devsim.add_1d_mesh_line(mesh=MESH,
                             pos=x_fine_end + i * dx_bulk,
                             ps=dx_bulk, tag=f"b{i}")

# FIX #6: contacts require tag=, not pos=/region=
devsim.add_1d_contact(mesh=MESH, name=SCK, tag="f0",        material="metal")
devsim.add_1d_contact(mesh=MESH, name=OHM, tag=f"b{Nbulk}", material="metal")
devsim.add_1d_region (mesh=MESH, material="Si", region=REG,
                      tag1="f0", tag2=f"b{Nbulk}")

devsim.finalize_mesh(mesh=MESH)
devsim.create_device(mesh=MESH, device=DEV)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PARAMETERS ON DEVICE
# ─────────────────────────────────────────────────────────────────────────────
_params = {
    "q": q, "k_B": k_B, "T": T, "V_T": V_T,
    "eps": eps_Si,
    "N_C": N_C, "N_V": N_V, "n_i": n_i,
    "mu_n": mu_n, "mu_p": mu_p, "D_n": D_n, "D_p": D_p,
    "N_D": N_D, "N_A": N_A,
    "phi_B": phi_B, "v_n": v_n, "n_s0": n_s0, "p_s0": p_s0,
    "tau_n": tau_n, "tau_p": tau_p,
    "V_schottky": 0.0, "V_ohmic": 0.0,
}
for name, val in _params.items():
    devsim.set_parameter(device=DEV, region=REG, name=name, value=val)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  SOLUTION VARIABLES  (FIX #3: must be declared before equations)
# ─────────────────────────────────────────────────────────────────────────────
devsim.node_solution(device=DEV, region=REG, name="Potential")
devsim.node_solution(device=DEV, region=REG, name="Electrons")
devsim.node_solution(device=DEV, region=REG, name="Holes")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  NODE MODELS  (FIX #4: Poisson charge + SRH with full Jacobian)
# ─────────────────────────────────────────────────────────────────────────────
# Net charge density ρ = q(p − n + N_D)  [C/cm³]
devsim.node_model(device=DEV, region=REG,
    name="charge",
    equation="q * (Holes - Electrons + N_D - N_A)")

# SRH recombination rate  U [cm⁻³ s⁻¹]
devsim.node_model(device=DEV, region=REG,
    name="SRH",
    equation=(
        "(Electrons * Holes - n_i^2)"
        " / (tau_p * (Electrons + n_i) + tau_n * (Holes + n_i))"
    ))

devsim.node_model(device=DEV, region=REG,
    name="SRH:Electrons",
    equation=(
        "(Holes*(tau_p*(Electrons+n_i)+tau_n*(Holes+n_i))"
        " - (Electrons*Holes-n_i^2)*tau_p)"
        " / (tau_p*(Electrons+n_i)+tau_n*(Holes+n_i))^2"
    ))

devsim.node_model(device=DEV, region=REG,
    name="SRH:Holes",
    equation=(
        "(Electrons*(tau_p*(Electrons+n_i)+tau_n*(Holes+n_i))"
        " - (Electrons*Holes-n_i^2)*tau_n)"
        " / (tau_p*(Electrons+n_i)+tau_n*(Holes+n_i))^2"
    ))

# ─────────────────────────────────────────────────────────────────────────────
# 11.  EDGE MODELS  (FIX #4 + FIX #5: SG currents with explicit Bernoulli)
# ─────────────────────────────────────────────────────────────────────────────

# Normalised potential difference across each mesh edge
devsim.edge_model(device=DEV, region=REG,
    name="vdiff",
    equation="(Potential@n1 - Potential@n0) / V_T")

# FIX #5: DEVSIM's SYMDIFF has NO bernoulli() built-in.
# We define B(x) = x/(exp(x)−1) explicitly with an ifelse guard:
#   |x| < 1e-4  →  Taylor: 1 − x/2 + x²/12   (avoids 0/0)
#   otherwise   →  exact formula
devsim.edge_model(device=DEV, region=REG,
    name="Bern01",
    equation=(
        "ifelse(abs(vdiff) < 1e-4,"
        "  1 - vdiff/2 + vdiff^2/12,"
        "  vdiff / (exp(vdiff) - 1))"
    ))

devsim.edge_model(device=DEV, region=REG,
    name="Bern10",
    equation=(
        "ifelse(abs(vdiff) < 1e-4,"
        "  1 + vdiff/2 + vdiff^2/12,"
        "  -vdiff / (exp(-vdiff) - 1))"
    ))

# Scharfetter-Gummel electron current  J_n  [A/cm²]
devsim.edge_model(device=DEV, region=REG,
    name="Jn",
    equation="q * D_n / EdgeLength * (Bern10 * Electrons@n1 - Bern01 * Electrons@n0)")
devsim.edge_model(device=DEV, region=REG,
    name="Jn:Electrons@n0",
    equation="-q * D_n / EdgeLength * Bern01")
devsim.edge_model(device=DEV, region=REG,
    name="Jn:Electrons@n1",
    equation="q * D_n / EdgeLength * Bern10")

# Scharfetter-Gummel hole current  J_p  [A/cm²]
devsim.edge_model(device=DEV, region=REG,
    name="Jp",
    equation="-q * D_p / EdgeLength * (Bern01 * Holes@n1 - Bern10 * Holes@n0)")
devsim.edge_model(device=DEV, region=REG,
    name="Jp:Holes@n0",
    equation="q * D_p / EdgeLength * Bern10")
devsim.edge_model(device=DEV, region=REG,
    name="Jp:Holes@n1",
    equation="-q * D_p / EdgeLength * Bern01")

# Poisson displacement flux  ε·∇φ  [C/cm²·s]
devsim.edge_model(device=DEV, region=REG,
    name="PotentialEdgeFlux",
    equation="eps * (Potential@n1 - Potential@n0) / EdgeLength")
devsim.edge_model(device=DEV, region=REG,
    name="PotentialEdgeFlux:Potential@n0",
    equation="-eps / EdgeLength")
devsim.edge_model(device=DEV, region=REG,
    name="PotentialEdgeFlux:Potential@n1",
    equation="eps / EdgeLength")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  GOVERNING EQUATIONS  (FIX #2 + #4: Poisson + continuity)
# ─────────────────────────────────────────────────────────────────────────────
devsim.equation(device=DEV, region=REG,
    name="PoissonEquation",
    variable_name="Potential",
    node_model="charge",
    edge_model="PotentialEdgeFlux",
    variable_update="log_damp")

devsim.equation(device=DEV, region=REG,
    name="ElectronContinuity",
    variable_name="Electrons",
    node_model="SRH",
    edge_model="Jn",
    variable_update="positive")

devsim.equation(device=DEV, region=REG,
    name="HoleContinuity",
    variable_name="Holes",
    node_model="SRH",
    edge_model="Jp",
    variable_update="positive")

# ─────────────────────────────────────────────────────────────────────────────
# 13.  BOUNDARY CONDITIONS  (FIX #5 + #6: thermionic + ohmic)
# ─────────────────────────────────────────────────────────────────────────────

# ── Schottky contact (anode, x = 0) ─────────────────────────────────────────
# Potential fixed to applied bias
devsim.contact_node_model(device=DEV, contact=SCK,
    name="SchottkyPotentialBC",
    equation="Potential - V_schottky")
devsim.contact_node_model(device=DEV, contact=SCK,
    name="SchottkyPotentialBC:Potential",
    equation="1")
devsim.contact_equation(device=DEV, contact=SCK,
    name="PoissonEquation",
    node_model="SchottkyPotentialBC",
    edge_model="")

# Thermionic emission:  J_n = q·v_n·(n − n_s0)
devsim.contact_node_model(device=DEV, contact=SCK,
    name="SchottkyElectronBC",
    equation="q * v_n * (Electrons - n_s0)")
devsim.contact_node_model(device=DEV, contact=SCK,
    name="SchottkyElectronBC:Electrons",
    equation="q * v_n")
devsim.contact_equation(device=DEV, contact=SCK,
    name="ElectronContinuity",
    node_model="SchottkyElectronBC",
    edge_model="")

# Holes pinned to equilibrium (metal = infinite recombination sink)
devsim.contact_node_model(device=DEV, contact=SCK,
    name="SchottkyHoleBC",
    equation="Holes - p_s0")
devsim.contact_node_model(device=DEV, contact=SCK,
    name="SchottkyHoleBC:Holes",
    equation="1")
devsim.contact_equation(device=DEV, contact=SCK,
    name="HoleContinuity",
    node_model="SchottkyHoleBC",
    edge_model="")

# ── Ohmic contact (cathode, x = L) ──────────────────────────────────────────
# Charge-neutral equilibrium:  n = N_D,  p = ni²/N_D
devsim.contact_node_model(device=DEV, contact=OHM,
    name="OhmicPotentialBC",
    equation=f"Potential - ({phi_back}) - V_ohmic")
devsim.contact_node_model(device=DEV, contact=OHM,
    name="OhmicPotentialBC:Potential",
    equation="1")
devsim.contact_equation(device=DEV, contact=OHM,
    name="PoissonEquation",
    node_model="OhmicPotentialBC",
    edge_model="")

devsim.contact_node_model(device=DEV, contact=OHM,
    name="OhmicElectronBC",
    equation=f"Electrons - {n_eq_back}")
devsim.contact_node_model(device=DEV, contact=OHM,
    name="OhmicElectronBC:Electrons",
    equation="1")
devsim.contact_equation(device=DEV, contact=OHM,
    name="ElectronContinuity",
    node_model="OhmicElectronBC",
    edge_model="")

devsim.contact_node_model(device=DEV, contact=OHM,
    name="OhmicHoleBC",
    equation=f"Holes - {p_eq_back}")
devsim.contact_node_model(device=DEV, contact=OHM,
    name="OhmicHoleBC:Holes",
    equation="1")
devsim.contact_equation(device=DEV, contact=OHM,
    name="HoleContinuity",
    node_model="OhmicHoleBC",
    edge_model="")

# ─────────────────────────────────────────────────────────────────────────────
# 14.  INITIAL CONDITIONS  (FIX #7: warm-start from bulk quasi-neutral guess)
#      Starting from zero causes the Newton solver to diverge because the
#      exponential terms (exp(φ/V_T)) blow up. Seeding with the charge-neutral
#      bulk solution puts us in the basin of convergence immediately.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 14.  INITIAL CONDITIONS
#      In DEVSIM, solution variable @n0/@n1 edge accessors are only populated
#      when the solution is initialised via init_from= pointing to an already-
#      evaluated node model.  The values= keyword writes to the node-model
#      layer but does NOT populate the internal solution vector that edge
#      model @-accessors read from.
#
#      Correct sequence:
#        (a) Register a constant node model for each variable
#        (b) Force-evaluate it with get_node_model_values — this materialises
#            the model's value array inside DEVSIM's memory
#        (c) Call set_node_values(init_from=<model_name>) to copy that array
#            into the solution variable
# ─────────────────────────────────────────────────────────────────────────────

# (a) Constant node models with bulk equilibrium values
devsim.node_model(device=DEV, region=REG,
    name="init_Potential", equation=str(float(phi_back)))
devsim.node_model(device=DEV, region=REG,
    name="init_Electrons", equation=str(float(n_eq_back)))
devsim.node_model(device=DEV, region=REG,
    name="init_Holes",     equation=str(float(p_eq_back)))

# (b) Force evaluation — materialises the model arrays in DEVSIM's cache
_phi_init = devsim.get_node_model_values(device=DEV, region=REG, name="init_Potential")
_n_init   = devsim.get_node_model_values(device=DEV, region=REG, name="init_Electrons")
_p_init   = devsim.get_node_model_values(device=DEV, region=REG, name="init_Holes")

# (c) Copy evaluated model arrays into the solution variables
devsim.set_node_values(device=DEV, region=REG,
    name="Potential", init_from="init_Potential")
devsim.set_node_values(device=DEV, region=REG,
    name="Electrons", init_from="init_Electrons")
devsim.set_node_values(device=DEV, region=REG,
    name="Holes",     init_from="init_Holes")

print(f"[init] {len(_phi_init)} nodes — "
      f"φ={_phi_init[0]:.4f} V, "
      f"n={_n_init[0]:.3e}, "
      f"p={_p_init[0]:.3e} cm⁻³")

# ─────────────────────────────────────────────────────────────────────────────
# 15.  EQUILIBRIUM SOLVE  (V = 0, establishes depletion layer)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Step 1:  Equilibrium (V = 0 V)")
print("="*60)
devsim.solve(type="dc",
             absolute_error=1e10,    # loose absolute tol — units are carriers/cm³
             relative_error=1e-10,
             maximum_iterations=30)
print("  Equilibrium converged.")

# ─────────────────────────────────────────────────────────────────────────────
# 16.  VOLTAGE SWEEP  (FIX #8 + #11)
#
#      FIX #8: Original code swept 0 → 1 V in 50 equal steps.
#        •  1 V forward bias is deep into high-injection — huge current,
#           very stiff system, guaranteed to diverge on large steps.
#        •  No reverse bias at all (misses the rectification behaviour).
#        •  Steps of 20 mV ≈ 0.77·V_T near threshold  → too coarse for
#           exponentially varying carrier concentrations.
#
#      Fixed sweep:
#        Reverse:  0 → −0.5 V   (25 steps of 20 mV)
#        Forward:  0 → +0.4 V   (80 steps of 5 mV — dense, exp region)
#
#      FIX #11: get_contact_current() requires the equation= argument.
#        Original call:  get_contact_current(device=, contact=)  → error
#        Fixed call:     get_contact_current(device=, contact=,
#                                            equation="ElectronContinuity")
# ─────────────────────────────────────────────────────────────────────────────
device_area = 1.0e-4     # 100 μm × 100 μm test pad [cm²]

V_sweep = np.sort(np.unique(np.concatenate([
    np.linspace( 0.00, -0.50, 26),   # reverse bias
    np.linspace( 0.01,  0.40, 80),   # forward bias (dense)
])))

print("\n" + "="*60)
print(f"Step 2:  I-V sweep  ({V_sweep[0]:.2f} V → {V_sweep[-1]:.2f} V,"
      f"  {len(V_sweep)} points)")
print("="*60)

results = []
for V_app in V_sweep:
    devsim.set_parameter(device=DEV, region=REG,
                         name="V_schottky", value=float(V_app))
    try:
        devsim.solve(type="dc",
                     absolute_error=1e10,
                     relative_error=1e-10,
                     maximum_iterations=30)

        # FIX #11: equation= argument is mandatory
        J = devsim.get_contact_current(device=DEV, contact=SCK,
                                        equation="ElectronContinuity")
        I = float(J) * device_area
        results.append((float(V_app), I))
        print(f"  V = {V_app:+.4f} V    I = {I:+.4e} A")

    except RuntimeError as err:
        print(f"  V = {V_app:+.4f} V    [convergence failure — skipped]")
        print(f"    ({err})")

# ─────────────────────────────────────────────────────────────────────────────
# 17.  SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
out_file = "schottky_IV.csv"
with open(out_file, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["V [V]", "I [A]", "J [A/cm2]", "log10|J|"])
    for V, I in results:
        J    = I / device_area
        logJ = math.log10(abs(J)) if abs(J) > 0 else -99
        w.writerow([V, I, J, logJ])

print(f"\n  Saved {len(results)} points → {out_file}")
print("\n" + "="*60)
print("Simulation complete.")
print("="*60)