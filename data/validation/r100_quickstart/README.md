# R=100 local injection-recovery reference

These compact reports record representative successful runs of the
new-user R=100 validation notebook. Both cases use 18 bins from 1.10 to
1.70 microns, 60 ppm Gaussian uncertainties, the bundled ExoMolOP H2O table,
60 MultiNest live points, and two MPI ranks.

The forward model is used directly to create the synthetic observation. Each
retrieval fits only `log_H2O`: emission uses a fixed non-isothermal temperature
profile and transmission uses a fixed reference radius. This avoids presenting
an intentionally under-constrained degeneracy as a successful installation
test.

The reports are reference outcomes, not bitwise golden files. A new run passes
when MultiNest converges, the injected abundance lies inside the posterior 95%
credible interval, and the reduced chi-square lies between 0.35 and 1.90.

Run `examples/notebooks/r100_emission_transmission_validation.ipynb` from top
to bottom to generate the forward models, observations, retrieval products,
plots, and fresh reports.
