# Blackbody Reference Curves

ROBERT uses blackbody curves as diagnostic references for emission spectra.
They are useful for plotting and scale checks, but they are not a substitute
for atmospheric radiative transfer.

The wavelength form of Planck's law is:

```text
B_lambda(lambda, T) = 2 h c^2 / (lambda^5 [exp(h c / (lambda k_B T)) - 1])
```

For a secondary-eclipse depth sanity check, ROBERT can compare two blackbodies:

```text
depth(lambda) = (B_planet(lambda) / B_star(lambda)) * (R_planet / R_star)^2
```

This estimate omits opacity, pressure-temperature structure, stellar atmosphere
models, instrument response, and retrieval parameters. Its purpose is to reveal
whether a candidate emission spectrum is in the right approximate visual range
before validated opacity and radiative-transfer components are available.
