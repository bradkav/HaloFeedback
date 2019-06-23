## Halo Feedback

Code for evolving a Dark Matter minispike under the influence of a perturbing body, injecting energy through dynamical friction.

#### Usage
```python
    def dfdt(self, r0, v_orb, v_cut=-1, method = 2, n_kick = 1):
    """
        Time derivative of the distribution function f(eps).
        
        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
            - method: optional, method = 1 or method = 2 chooses two different
                        approximate calculations of the scattering probability
                        (method 2 should be more accurate)
            - n_kick: optional, number of 'points' to include in the numerical
                    integration over delta-eps. Default: n_kick = 1, which
                    simply uses the average value of delta-eps.
    """
```

#### Updates

**12/06/2019 (a):** The class DistributionFunction() now takes an optional argument `Lambda` when initialising. If you don't specify `Lambda`, it's calculated as `Sqrt(M_BH/M_NS)`.  
**12/06/2019 (b):** Updated integration over delta-eps (now uses a correct trapezoidal scheme). May be a bit slower, but no change needed by the user.  
**19/06/2019:** Updated the method *yet again*. Basically, I've reverted to the old method of using a single 'kick' in energy, which seems to conserve energy correctly.  
**20/06/2019:** I've now added a new method, which does the calculation a bit more carefully - unfortunately we're still not conserving energy :(  
**23/06/2019:** Added "average" option to dynamical friction calculation (allowing you to average the density over r0 - b_max < r < r0 + b_max before calculating the DF force). Energy should be conserved at the %-level or better now.
