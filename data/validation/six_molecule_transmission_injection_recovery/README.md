# Six-Molecule Transmission Injection Recovery

This validation snapshot records a deterministic configured retrieval of H2O,
CO, CO2, CH4, NH3, HCN, and reference-radius scale from a synthetic clear
transmission spectrum. It uses the real ExoMolOP R=15000 POKAZATEL, Li2015,
UCL-4000, YT34to10, CoYuTe, and Harris cross-section databases.

The injection has 96 logarithmically spaced bins from 0.6 to 12 micron with
independent 8 ppm Gaussian noise. The forward model uses 48 pressure layers,
8-point correlated-k tables, random overlap, H2/He CIA and Rayleigh extinction,
inverse-square gravity, and sixth-order impact quadrature. MultiNest ran with
2 MPI processes, 40 live points, and `dlogz=0.5`.

All seven parameters passed their preregistered absolute recovery tolerances.
The largest molecular error was 0.062 dex for HCN; the reference-radius scale
error was 0.000160. The best fit has reduced chi-square 0.941 for 96 points and
7 fitted parameters. Sampling converged in 151 seconds, or 164 seconds including
configured setup and result handling, after 8705 likelihood evaluations.

This is a same-model closure test. It validates YAML integration, real-opacity
preparation, six-gas random overlap, radius retrieval, MultiNest orchestration,
and post-processing. It does not independently validate the physics; the
stable-petitRADTRANS comparison in the adjacent validation snapshot provides
that external forward-model check.

Reproduce with:

```bash
/Users/jaketaylor/miniforge3/envs/robert-exoplanets/bin/python \
  examples/synthetic_six_molecule_transmission_injection_recovery.py \
  --config configurations/synthetic_six_molecule_transmission_injection_recovery_multinest.yaml

/Users/jaketaylor/miniforge3/envs/robert-exoplanets/bin/mpirun -np 2 \
  /Users/jaketaylor/miniforge3/envs/robert-exoplanets/bin/python \
  run_retrieval.py \
  --config configurations/synthetic_six_molecule_transmission_injection_recovery_multinest.yaml

/Users/jaketaylor/miniforge3/envs/robert-exoplanets/bin/python \
  examples/synthetic_six_molecule_transmission_injection_recovery.py \
  --config configurations/synthetic_six_molecule_transmission_injection_recovery_multinest.yaml \
  --evaluate-result \
  --validation-dir data/validation/six_molecule_transmission_injection_recovery
```
