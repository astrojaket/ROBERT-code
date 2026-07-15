# Multispecies Transmission Validation

This snapshot compares ROBERT with stable petitRADTRANS 3.3.3 for an 80-layer,
0.3--12 micron atmosphere containing H2O, CO, CO2, CH4, NH3, HCN, H2-H2 and
H2-He CIA, and H2/He Rayleigh extinction.

The independent pRT run evaluates the six native opacity tables and exports
species-resolved evaluated opacities. ROBERT converts those values to effective
molecular cross-sections, combines the species using random overlap, constructs
hydrostatic spherical geometry, and solves the transmission spectrum. This
isolates mixture, geometry, and radiative-transfer behavior from opacity-table
and interpolation differences.

The shared-Rayleigh comparison is the strict solver gate. ROBERT's native
Rayleigh comparison additionally tests its independent H2/He cross-section
implementation. Timing compares steady-state forward calculations after table
loading; because the two codes do not perform identical work, it is an
engineering performance indicator rather than a microbenchmark.

The JSON report also records the pressure-radius anchors and a diagnostic
constant-radius alignment. The raw comparison remains primary; alignment
separates a retrievable absolute-radius mode from spectral-shape differences.

See `docs/review/19_petitradtrans3_multispecies_transmission.md` for the physical
contract, results, limitations, and reproduction commands.
