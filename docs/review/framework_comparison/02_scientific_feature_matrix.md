# Scientific Feature Matrix

Legend: `Y` = supported in inspected source or primary docs, `P` = partial,
experimental, or indirect support, `N` = no clear support in inspected source,
`U` = unclear from available source.

## Spectral Modes and Geometries

| Feature | NEMESIS | NemesisPy | TauREx | POSEIDON | petitRADTRANS | PICASO | CHIMERA | Brewster | Exo_Skryer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Emission spectroscopy | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Transmission spectroscopy | Y | Y | Y | Y | Y | Y | Y | N | Y |
| Reflection spectroscopy | Y | P | N/P | Y | P | Y | Y | N/P | Y |
| Eclipse modeling | P | Y | P | Y | Y | P | Y | P | Y |
| Phase curves | P | Y | P | P | P | Y | P | N | P |
| 1D atmospheres | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| 1.5D / patchy columns | P | P | P | Y | P | Y | P | Y | Y |
| 2D/3D atmospheres | P | Y | P | Y | P | Y | N/P | N | P |
| Direct imaging / brown dwarfs | P | P | P | Y | Y | Y | P | Y | Y |
| Polarisation | N/U | N | N | N/U | N/U | N/U | N/U | N/U | N/U |

## Radiative Transfer and Opacity

| Feature | NEMESIS | NemesisPy | TauREx | POSEIDON | petitRADTRANS | PICASO | CHIMERA | Brewster | Exo_Skryer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scattering | Y | P | P | Y | Y | Y | Y | Y | Y |
| Multiple scattering | Y | P | P | Y | Y | Y | Y | Y | P/Y |
| Line-by-line / opacity sampling | Y | P | Y | Y | Y | P | P | P | Y |
| Correlated-k | Y | Y | Y | Y | Y | Y | Y | P/Y | Y |
| CK mixing choices | Y | P | P | Y | Y | Y | Y | P | Y |
| CIA | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Rayleigh scattering | Y | Y | Y | Y | Y | Y | P | P | Y |
| H-minus / special continuum | P | P | P | Y | Y | P | P | Y | Y |
| Multiple opacity databases | Y | Y | Y | Y | Y | Y | Y | P | Y |
| Opacity cache / registry | P | P | Y | Y | Y | Y | P | P | Y |
| Contribution functions | Y | Y | Y | Y | Y | Y | Y | Y | Y |

## Chemistry and Clouds

| Feature | NEMESIS | NemesisPy | TauREx | POSEIDON | petitRADTRANS | PICASO | CHIMERA | Brewster | Exo_Skryer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Free chemistry | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Equilibrium chemistry | Y/P | P | Y | Y | Y | Y | Y | Y | Y |
| Disequilibrium / quench chemistry | P | P | P | Y | Y/P | Y | Y/P | P | Y |
| Photochemistry interface | N/U | N/U | P | P | P | P | N/U | N/U | N/U |
| Multiple chemistry backends | P | P | P | P | P | Y | P | P | Y |
| Gray clouds | Y | P | Y | Y | Y | Y | Y | Y | Y |
| Mie / microphysical clouds | Y | P | P | Y | Y | Y | Y | Y | Y |
| Patchy clouds | Y/P | P | P | Y | P | Y | P | Y | P |
| Ackerman-Marley style clouds | P | N/U | P | P | Y/P | Y | Y | P | N/U |
| Retrieved refractive index / n,k | N/U | N/U | N/U | P | P | P | N/U | P | Y |

## Stellar and Instrument Effects

| Feature | NEMESIS | NemesisPy | TauREx | POSEIDON | petitRADTRANS | PICASO | CHIMERA | Brewster | Exo_Skryer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stellar spectra | Y | Y | Y | Y | Y | Y | P | P | Y |
| Stellar contamination models | N/P | N/U | P | Y | P | P | P | N | N/P |
| Multi-instrument data | P | P | Y | Y | Y | Y | P | Y | Y |
| Instrument convolution/binning | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Instrument offsets | P | P | P | Y | Y | Y/P | P | Y | Y |
| High-resolution cross correlation | N/P | P | P | Y | Y | P | P | P | P |

## ROBERT Takeaways

- Must-have for ROBERT: emission spectroscopy, correlated-k, CIA, Rayleigh,
  basic clouds, contribution functions, and instrument binning.
- Nice-to-have after the core is stable: line-by-line/opacity-sampling mode,
  transmission, stellar contamination, patchy clouds, and multiple chemistry
  backends.
- Future-only unless a science case forces it: phase curves, full 3D, polarisation,
  broad photochemistry coupling, and GPU-specific optimization.
