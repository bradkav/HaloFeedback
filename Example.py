# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## HaloFeedback
#
# **Units:**
#     - Masses [M_sun]
#     - Times [s]
#     - Distances [pc]
#     - Speeds [km/s]
#     - Density [M_sun/pc^3]

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
import HaloFeedback

# %% [markdown]
# #### Initialisation
#
# Initialise a distribution function:

# %%
DF1 = HaloFeedback.PowerLawSpike()

# This defaults to some standard values for the
# density profile and masses:

print(DF1.gamma)
print(DF1.M_BH, " M_sun")
print(DF1.M_NS, " M_sun")
print(DF1.rho_sp, " M_sun/pc^3")

# %% [markdown]
# Otherwise, you can specify the values you're interested in:

# %%
DF2 = HaloFeedback.PowerLawSpike(M_BH=1e4, M_NS=1, gamma=2.5, rho_sp=226.0)

# %% [markdown]
# #### Extracting the density
#
# The density at a given radius is extracted with:

# %%
DF2.rho(1e-8)

# %% [markdown]
# This calculates the density numerically from the internally stored distribution function $f(\mathcal{E})$.
#
# If we want to calculate the density of particles at a given radius with speeds below some v_cut, then we can do:

# %%
DF2.rho(1e-8, v_cut=1e4)

# %% [markdown]
# #### Evolving the system
#
# The internal distribution function $f(\mathcal{E})$ is stored as a grid over $\mathcal{E}$ values:

# %%
DF2.eps_grid

# %%
DF2.f_eps

# %% [markdown]
# For a compact object at radius $r_0$ with velocity $v_0$, the time derivative of the distribution function can be calculated using:

# %%
r0 = 1e-8  # pc
v0 = 5e4  # km/s
DF2.dfdt(r0, v0)

# %% [markdown]
# Again, if we want to scatter only with particles with speed smaller than some v_cut, we can use:

# %%
DF2.dfdt(r0, v0, v_cut=2e4)

# %% [markdown]
# Note that typically v_cut will be equal to the orbital speed of the compact object.
#
# Finally, then, we update the internal distribution function using:

# %%
dt = 1  # second
DF2.f_eps += DF2.dfdt(r0, v0, v_cut=2e4) * dt
# In the 'EvolutionPlot.py' file, I evolve using the improved Euler method, which should be more stable...

# %% [markdown]
# Then we can recompute the density:

# %%
DF2.rho(r0)

# %%
