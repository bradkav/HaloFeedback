import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz

import HaloFeedback
from HaloFeedback import G_N, c

from matplotlib import gridspec
import matplotlib

from pathlib import Path

from tqdm import tqdm

USE_RK4 = True
REFINE = False

def evolve_density(M1, M2, x0, b_max_over_r0, N_orb, orbits_per_step = 1, orbits_per_output = 10):
    """
    Arguments:
        M1: Primary mass [Msun]
        M2: Secondary mass [Msun]
        x0: Orbital radius in units of r_isco(M1)
        b_max_over_r0: b_max in units of the orbital radius
        N_orb: number of orbits to simulate for
        orbits_per_step: Number of orbits per timestep
        orbits_per_output: number of orbits between output steps
    """
    
    
    r_isco = 6.0 * G_N * M1 / c ** 2
    
    r0 = x0*r_isco
    b_max = b_max_over_r0*r0

    v0 = np.sqrt(G_N * (M1 + M2) / (r0))
    v_cut = v0

    r_t = 20*r0
    
    DF = HaloFeedback.GeneralizedNFWSpike(M_BH=M1, M_NS=M2, r_t = r_t, rho_6=9.1e14, b_max_fixed=b_max)
    DF2 = HaloFeedback.GeneralizedNFWSpike(M_BH=M1, M_NS=M2, r_t = r_t, rho_6=9.1e14, b_max_fixed=b_max)

    if (orbits_per_output < orbits_per_step):
        print("> Warning: orbits_per_output < orbits_per_step. Setting orbits_per_output = orbits_per_step.")
        orbits_per_output = orbits_per_step
    
    # Orbital time in seconds
    T_orb = 2 * np.pi * r0 * 3.0857e13 / v0

    N_step = int(N_orb / orbits_per_step)
    N_step_out = int(N_orb / orbits_per_output)

    dt = T_orb * orbits_per_step

    t_list = dt * N_step * np.linspace(0, 1, N_step_out + 1)

    print("    Number of orbits:", N_orb)
    print("    Total Time [days]:", N_step * dt / (3600 * 24))

    # Radial grid for calculating the density
    N_r = 201
    r_list = np.geomspace(1e2, 1e6, N_r - 1)*r_isco
    r_list = np.sort(np.append(r_list, r0))

    # List of densities
    rho_list = np.zeros((N_step_out + 1, N_r))
    rhocut_list = np.zeros((N_step_out + 1, N_r))

    # Which index refers to r0?
    r0_ind = np.where(r_list == r0)[0][0]
    #print(r0_ind)

    # Rate of dynamical friction energy loss
    DF_list = np.zeros(N_step_out + 1)
    DF_list[0] = DF.dEdt_DF(r0, v_cut)

    # Initial density

    # If we're only interested in particles below the local orbital speed
    rho0_cut = np.array(
            [DF.rho(r, v_cut=np.sqrt(G_N * (DF.M_BH + DF.M_NS) / r)) for r in r_list]
        )
    rho0 = np.array([DF.rho(r) for r in r_list])
    
    rhocut_list[0, :] = rho0_cut
    rho_list[0, :] = rho0

    # ----------- Evolving the system ----------

    #print(N_step)
    output_inds = []
    for i in tqdm(range(N_step)):

        i_orb = i*orbits_per_step
        # Calculate the density profile at this timestep
        if (i_orb%orbits_per_output == 0):
            ind = int(i_orb/orbits_per_output)
            #print(i_orb, ind)
            #output_inds.append(ind)
            
            rhocut_list[ind, :] = np.array(
                [DF.rho(r, v_cut=np.sqrt(G_N * (DF.M_BH + DF.M_NS) / r)) for r in r_list]
            )
            rho_list[ind, :] = np.array([DF.rho(r) for r in r_list])

        if (not USE_RK4):
            # Time-step using the improved Euler method
            df1 = DF.delta_f(r0=r0, v_orb=v0, dt=dt, v_cut=v_cut)
            DF.f_eps += df1
            df2 = DF.delta_f(r0=r0, v_orb=v0, dt=dt, v_cut=v_cut)
            DF.f_eps += 0.5 * (df2 - df1)

        else:
            
            N_sub = 1
            
            if (REFINE):
            
                df_minus = DF.dfdt_minus(r0=r0, v_orb=v0, v_cut=v_cut, n_kick=HaloFeedback.N_KICK)
                dt_max = DF.f_eps/np.abs(df_minus)
            
                N_sub = int(np.ceil(dt/np.min(dt_max)))
                        
                if (N_sub > dt/T_orb):
                    N_sub = int(np.floor(dt/T_orb))
            
                if (N_sub > 1):
                    print("N_sub = ", N_sub)
            
            dt_sub = dt/N_sub
            
            for i_sub in range(N_sub):
        
                f0 = DF.f_eps

                k1 = DF.delta_f(r0=r0, v_orb=v0, dt=dt_sub, v_cut=v_cut)

                DF.f_eps = f0 + 0.5*k1
                k2 = DF.delta_f(r0=r0, v_orb=v0, dt=dt_sub, v_cut=v_cut)

                DF.f_eps = f0 + 0.5*k2
                k3 = DF.delta_f(r0=r0, v_orb=v0, dt=dt_sub, v_cut=v_cut)

                DF.f_eps = f0 + 0.5*k3
                k4 = DF.delta_f(r0=r0, v_orb=v0, dt=dt_sub, v_cut=v_cut)

                DF.f_eps = f0 + (k1 + 2*k2 + 2*k3 + k4)/6
    
    output_inds.append(-1)
    rhocut_list[-1, :] = np.array(
        [DF.rho(r, v_cut=np.sqrt(G_N * (DF.M_BH + DF.M_NS) / r)) for r in r_list]
    )
    rho_list[-1, :] = np.array([DF.rho(r) for r in r_list])

    folder_path = Path("output")
    folder_path.mkdir(parents=True, exist_ok=True)
    file_label = f"M1_{M1:.1f}_M2_{M2:.1f}_x0_{x0:.1f}_b_max_{b_max_over_r0:.3f}_Norb_{N_orb}_dN_{orbits_per_step}"  
    
    #Grid of radii
    np.savetxt(f"output/r_M1_{M1:.1f}.txt", r_list)
    
    #Density on the radial grid, at each output time
    np.savetxt(f"output/Density_grid_{file_label}.txt", (rho_list/np.atleast_2d(rho0)).T)
    
    #Density (v < v_orb) on the radial grid, at each output time
    np.savetxt(f"output/Density_grid_vcut_{file_label}.txt", (rhocut_list/np.atleast_2d(rho0_cut)).T)
    
    #Density at the orbital radius as a function of time
    np.savetxt(f"output/Density_{file_label}.txt", np.c_[t_list/T_orb, rho_list[:,r0_ind], rhocut_list[:,r0_ind]])


mode = "test"

if (mode == "test"):
    risco = 6.0 * G_N * 1400 / c ** 2
    b_max_over_r0 = np.sqrt(1/1000)
    evolve_density(1400, 1.4, x0=1e-8/risco, b_max_over_r0=b_max_over_r0, N_orb=40000, orbits_per_step = 100, orbits_per_output=200)
