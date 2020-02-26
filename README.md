## Halo Feedback

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for evolving a Dark Matter minispike under the influence of a perturbing body, injecting energy through dynamical friction.

#### Getting started

See the example notebook [`Example.ipynb`](https://github.com/bradkav/HaloFeedback/blob/master/Example.ipynb) on how to initialise the code and the DM halo. The script [`PlotEvolution.py`](https://github.com/bradkav/HaloFeedback/blob/master/PlotEvolution.py) evolves the halo in the case of an body orbiting at fixed radius, and generates various illustrative plots.

#### Usage

The main functionality of the code is through the function `dfdt`, which calculates the time derivative of the distribution function over a grid of energies Epsilon.

```python
    def dfdt(self, r0, v_orb, v_cut=-1):
    """
        Time derivative of the distribution function f(eps).
        
        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity of the pertubing body [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
    """
```

An alternative implementation `delta_f` computes the change in the distribution function over a timestep `dt`. This behaves better under large timestep as it prevents the distribution function going negative.

```python
    def delta_f(self, r0, v_orb, dt, v_cut=-1):
        """Change in f over a time-step dt.
        Automatically prevents f_eps going below zero.       
        
        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - dt: time-step [s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
        """

```

The code also allows you to calculate the rate at which energy is carried away from the system. Particles which receive a 'kick' in energy that unbinds them (Epsilon < 0) are no longer tracked by the code, so this function allows you to calculate how much energy is being carried away by these particles are each timestep, in order to check energy conservation.

```python
    def dEdt_ej(self, r0, v_orb, v_cut=-1):
    """
        Calculate carried away by particles which are completely unbound.
        
        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
            - n_kick: optional, number of grid points to use when integrating over
                        Delta-eps (defaults to N_KICK = 100).
    """
```

At any time during the simulation, the density of DM at a given radius can be extracted using:

```python

    def rho(self, r, v_cut=-1):
    """
        DM mass density computed from f(eps).
        
        Parameters: 
            - r : radius in pc
            - v_cut : maximum speed to include in density calculation
                     (defaults to v_max if not specified)
    """

```

#### Updates

- *26/02/2020*:  Cleanup up `PlotEvolution.py` script. Ready for initial realise alongside arXiv preprint.  
- *02/12/2019:** Rewritten the code to do the calculation more carefully, in particular integrating over the different sizes of kick (Delta-epsilon) now conserves energy. The full integration over Delta-epsilon is now the default option (in fact, using a single average kick is no longer supported, although I might bring it back later.) Note  also that the "average = True" option for the dynamical friction calculation is now *no longer recommended*.  
 - *23/06/2019:* Added "average" option to dynamical friction calculation (allowing you to average the density over r0 - b_max < r < r0 + b_max before calculating the DF force). Energy should be conserved at the %-level or better now.  
 - *20/06/2019:* I've now added a new method, which does the calculation a bit more carefully - unfortunately we're still not conserving energy :(  
 - *19/06/2019:* Updated the method *yet again*. Basically, I've reverted to the old method of using a single 'kick' in energy, which seems to conserve energy correctly.  
 - *12/06/2019 (b):* Updated integration over delta-eps (now uses a correct trapezoidal scheme). May be a bit slower, but no change needed by the user.  
 - *12/06/2019 (a):* The class DistributionFunction() now takes an optional argument `Lambda` when initialising. If you don't specify `Lambda`, it's calculated as `Sqrt(M_BH/M_NS)`.  

#### License

This work is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

