## Halo Feedback

Code for evolving a Dark Matter minispike under the influence of a perturbing body, injecting energy through dynamical friction.

#### Updates

**12/06/2019 (a):** The class DistributionFunction() now takes an optional argument `Lambda` when initialising. If you don't specify `Lambda`, it's calculated as `Sqrt(M_BH/M_NS)`.
**12/06/2019 (b):** Updated integration over delta-eps (now uses a correct trapezoidal scheme). May be a bit slower, but not change needed by the user.  
**19/06/2019:** Updated the method *yet again*. Basically, I've reverted to the old method of using a single 'kick' in energy, which seems to conserve energy correctly.