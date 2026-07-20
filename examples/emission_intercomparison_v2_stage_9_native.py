"""Persistent native forward adapters for emission intercomparison V2 Stage 9.

Each adapter is instantiated once per MPI rank and reuses its native opacity
objects across likelihood calls.  The only common numerical operations after a
native framework returns are the frozen R=100 flux-conserving binning and the
common eclipse-depth normalization.  No shared opacity or cloud tensor crosses
framework boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from robert_exoplanets import (
    ParmentierGuillot2014TemperatureProfile,
    PressureGrid,
    Spectrum,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2 import (
    flux_conserving_bin_mean,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_7 import (
    power_law_cloud_tau,
)
from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (
    scenario_by_name,
)


MOLECULAR_SPECIES = ("H2O", "CO", "CO2", "CH4")
PRT_LINE_SPECIES = {
    "H2O": "H2O__POKAZATEL",
    "CO": "CO__HITEMP",
    "CO2": "CO2__UCL-4000",
    "CH4": "CH4__YT34to10",
}
PRT_CIA_SPECIES = {
    "H2-H2": "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
    "H2-He": "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
}


def _pressure_grid(common: Mapping[str, Any]) -> PressureGrid:
    selected = next(item for item in common["pressure_grids"] if item["n_cells"] == 80)
    return PressureGrid(
        edges=np.asarray(selected["edges_bar"], dtype=float),
        centers=np.asarray(selected["centers_bar"], dtype=float),
        unit="bar",
        name="emission-v2-stage9-80-cell",
    )


def _edge_grid(pressure: PressureGrid) -> PressureGrid:
    return PressureGrid.from_log_centers(
        float(pressure.edges[0]),
        float(pressure.edges[-1]),
        pressure.edges.size,
        unit="bar",
        name="emission-v2-stage9-level-temperature",
    )


@dataclass(frozen=True)
class AtmosphericState:
    temperature_cells_k: NDArray[np.float64]
    temperature_edges_k: NDArray[np.float64]
    gas_names: tuple[str, ...]
    gas_vmr: NDArray[np.float64]
    gas_mass_u: NDArray[np.float64]
    mean_molecular_weight_u: float
    cloud_tau_5um: float
    cloud_top_pressure_bar: float
    cloud_single_scattering_albedo: float


def atmospheric_state(
    common: Mapping[str, Any], scenario_name: str, values: Mapping[str, float]
) -> AtmosphericState:
    """Map retrieval coordinates to the exact common atmosphere arrays."""

    scenario = scenario_by_name(scenario_name)
    pressure = _pressure_grid(common)
    profile = ParmentierGuillot2014TemperatureProfile(
        gravity=float(common["derived_quantities"]["surface_gravity_m_s2"]),
        internal_temperature=100.0,
        kappa_ir_parameter_name="kappa_IR",
        gamma1_parameter_name="gamma1",
        gamma2_parameter_name="gamma2",
        irradiation_temperature_parameter_name="T_irr",
        alpha_parameter_name="alpha",
    )
    linear = {
        "kappa_IR": 10.0 ** float(values["log10_kappa_ir"]),
        "gamma1": 10.0 ** float(values["log10_gamma1"]),
        "gamma2": 10.0 ** float(values["log10_gamma2"]),
        "T_irr": float(values["T_irr_k"]),
        "alpha": float(values["alpha"]),
    }
    temperature_cells = profile.evaluate(linear, pressure)
    temperature_edges = profile.evaluate(linear, _edge_grid(pressure))

    active = {
        name: 10.0 ** float(values[f"log10_vmr_{name}"]) for name in MOLECULAR_SPECIES
    }
    active_sum = sum(active.values())
    if active_sum >= 1.0:
        raise ValueError("active gas VMRs must sum to less than one")
    remainder = 1.0 - active_sum
    composition = {
        "H2": remainder * 0.8547,
        "He": remainder * 0.1453,
        **active,
    }
    names = tuple(composition)
    vmr = np.asarray([composition[name] for name in names], dtype=float)
    masses_by_name = common["composition"]["molecular_masses_u"]
    masses = np.asarray([masses_by_name[name] for name in names], dtype=float)
    mmw = float(np.sum(vmr * masses))
    if scenario.cloudy:
        cloud_tau = 10.0 ** float(values["log10_cloud_tau_5um"])
        cloud_top = 10.0 ** float(values["log10_cloud_top_pressure_bar"])
    else:
        cloud_tau = 0.0
        cloud_top = 1.0e-2
    omega = (
        float(values["cloud_single_scattering_albedo"])
        if scenario.cloud == "grey_isotropic_scattering"
        else 0.0
    )
    return AtmosphericState(
        temperature_cells,
        temperature_edges,
        names,
        vmr,
        masses,
        mmw,
        cloud_tau,
        cloud_top,
        omega,
    )


def truth_parameters(common: Mapping[str, Any], scenario_name: str) -> dict[str, float]:
    """Return the exact truth vector encoded in the frozen common contract."""

    from robert_exoplanets.diagnostics.emission_intercomparison_v2_stage_9 import (
        parameter_definitions,
    )

    return {item.name: item.truth for item in parameter_definitions(scenario_name)}


def _close_edges(
    wavelength: NDArray[np.float64],
    flux: NDArray[np.float64],
    edges: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(flux, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x[0] > edges[0]:
        x = np.concatenate(([edges[0]], x))
        y = np.concatenate((y[:1], y))
    if x[-1] < edges[-1]:
        x = np.concatenate((x, [edges[-1]]))
        y = np.concatenate((y, y[-1:]))
    return x, y


def _native_bin_overlap_mean(
    native_lower_micron: NDArray[np.float64],
    native_upper_micron: NDArray[np.float64],
    values: NDArray[np.float64],
    target_edges_micron: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Average native spectral-density bins by their physical overlap."""

    lower = np.asarray(native_lower_micron, dtype=float)
    upper = np.asarray(native_upper_micron, dtype=float)
    spectra = np.asarray(values, dtype=float)
    targets = np.asarray(target_edges_micron, dtype=float)
    if (
        lower.ndim != 1
        or upper.shape != lower.shape
        or spectra.shape[-1] != lower.size
        or targets.ndim != 1
    ):
        raise ValueError("native-bin projection axes are inconsistent")
    if (
        np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(upper <= lower)
        or np.any(np.diff(lower) <= 0.0)
        or np.any(np.diff(targets) <= 0.0)
    ):
        raise ValueError("native-bin projection requires finite ordered supports")
    scale = max(float(np.max(np.abs(targets))), 1.0)
    tolerance = 1.0e-10 * scale
    if np.any(lower[1:] < upper[:-1] - tolerance):
        raise ValueError("native spectral-bin supports overlap")

    flat = spectra.reshape((-1, lower.size))
    output = np.empty((flat.shape[0], targets.size - 1), dtype=float)
    for index, (target_lower, target_upper) in enumerate(
        zip(targets[:-1], targets[1:], strict=True)
    ):
        overlap = np.maximum(
            0.0,
            np.minimum(upper, target_upper) - np.maximum(lower, target_lower),
        )
        covered = float(np.sum(overlap))
        target_width = float(target_upper - target_lower)
        if not np.isclose(covered, target_width, rtol=2.0e-6, atol=tolerance):
            raise ValueError(
                "PICASO native bins do not fully and uniquely cover an R=100 bin"
            )
        output[:, index] = np.sum(flat * overlap[None, :], axis=1) / covered
    return output.reshape((*spectra.shape[:-1], output.shape[-1]))


def _picaso_native_wavelength_support(
    wavenumber_cm1: NDArray[np.float64],
    delta_wavenumber_cm1: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Convert PICASO native wavenumber centres and widths to bin supports."""

    wavenumber = np.asarray(wavenumber_cm1, dtype=float)
    widths = np.asarray(delta_wavenumber_cm1, dtype=float)
    if (
        wavenumber.ndim != 1
        or widths.shape != wavenumber.shape
        or np.any(~np.isfinite(wavenumber))
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0.0)
        or np.any(wavenumber - 0.5 * widths <= 0.0)
    ):
        raise ValueError("invalid PICASO native wavenumber-bin definition")
    wavenumber_lower = wavenumber - 0.5 * widths
    wavenumber_upper = wavenumber + 0.5 * widths
    wavelength_lower = 1.0e4 / wavenumber_upper
    wavelength_upper = 1.0e4 / wavenumber_lower
    wavelength_center = 1.0e4 / wavenumber
    order = np.argsort(wavelength_center)
    return (
        wavelength_center[order],
        wavelength_lower[order],
        wavelength_upper[order],
        order.astype(np.int64),
    )


class NativeForward:
    """Base class for a persistent native Stage-9 forward model."""

    framework: str

    def __init__(self, common: Mapping[str, Any], scenario_name: str) -> None:
        self.common = common
        self.scenario = scenario_by_name(scenario_name)
        self.pressure = _pressure_grid(common)
        spectral = common["spectral_contract"]
        self.r100_centers = np.asarray(spectral["r100_centers_micron"], dtype=float)
        self.r100_edges = np.asarray(spectral["r100_edges_micron"], dtype=float)
        self.stellar_r100 = np.asarray(
            common["stellar_blackbody"]["r100_surface_flux_w_m2_m"], dtype=float
        )
        self.area_ratio = float(common["derived_quantities"]["projected_area_ratio"])
        self.last_native_wavelength = np.empty(0)
        self.last_native_flux = np.empty(0)
        self.last_native_bin_lower_micron = np.empty(0)
        self.last_native_bin_upper_micron = np.empty(0)
        self.native_binning_method = "piecewise_linear_center_integration"

    def native_flux(
        self, values: Mapping[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raise NotImplementedError

    def eclipse_depth(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        wavelength, native_flux = self.native_flux(values)
        wavelength, native_flux = _close_edges(wavelength, native_flux, self.r100_edges)
        binned = flux_conserving_bin_mean(wavelength, native_flux, self.r100_edges)
        self.last_native_wavelength = np.asarray(wavelength, dtype=float)
        self.last_native_flux = np.asarray(native_flux, dtype=float)
        return np.asarray(binned / self.stellar_r100 * self.area_ratio, dtype=float)

    def spectrum(self, values: Mapping[str, float]) -> Spectrum:
        return Spectrum.from_arrays(
            self.r100_centers,
            self.eclipse_depth(values),
            unit="eclipse_depth",
            observable="eclipse_depth",
        )


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one reference asset matching {pattern} below {root}"
        )
    return matches[0]


class RobertNativeForward(NativeForward):
    """ROBERT native RORR correlated-k/CIA and SH4 grey-cloud path."""

    framework = "robert"

    def __init__(self, common: Mapping[str, Any], scenario_name: str) -> None:
        super().__init__(common, scenario_name)
        from robert_exoplanets import (
            CiaTable,
            CorrelatedKOpacityProvider,
            CorrelatedKTable,
            PreparedCorrelatedKOpacity,
            SpectralGrid,
        )

        root = Path(os.environ["STAGE9_PRT_INPUT_DATA"]).expanduser().resolve()
        patterns = {
            "H2O": "*POKAZATEL*.ktable.petitRADTRANS.h5",
            "CO": "*HITEMP*.ktable.petitRADTRANS.h5",
            "CO2": "*UCL-4000*.ktable.petitRADTRANS.h5",
            "CH4": "*YT34to10*.ktable.petitRADTRANS.h5",
            "H2-H2": "*H2--H2*.ciatable.petitRADTRANS.h5",
            "H2-He": "*H2--He*.ciatable.petitRADTRANS.h5",
        }
        paths = {name: _find_one(root, pattern) for name, pattern in patterns.items()}
        self.tables = {
            name: CorrelatedKTable.from_petitradtrans_hdf(paths[name], species=name)
            for name in MOLECULAR_SPECIES
        }
        first = self.tables["H2O"]
        mask = (first.wavelength_micron >= 0.3) & (first.wavelength_micron <= 12.1)
        self.wavelength = np.sort(first.wavelength_micron[mask])
        spectral_grid = SpectralGrid.from_array(
            self.wavelength, unit="micron", role="opacity"
        )
        self.providers = {
            name: CorrelatedKOpacityProvider(
                {name: table},
                name=f"stage9-{name}",
                interpolation="log_pressure_temperature_log_k",
            )
            for name, table in self.tables.items()
        }
        self.prepared_species = {
            name: provider.prepare(spectral_grid, self.pressure, species=(name,))
            for name, provider in self.providers.items()
        }
        self.prepared = PreparedCorrelatedKOpacity(
            provider_name="stage9-prt-hdf-four-species",
            spectral_grid=spectral_grid,
            pressure_grid=self.pressure,
            species=MOLECULAR_SPECIES,
            g_samples=first.g_samples,
            g_weights=first.g_weights,
            cache_key="stage9-80",
            metadata={"gas_combination": "random_overlap"},
        )
        self.cia_tables = {
            pair: CiaTable.from_petitradtrans_hdf(paths[pair], collision_pair=pair)
            for pair in ("H2-H2", "H2-He")
        }
        self.g_weights = np.asarray(first.g_weights, dtype=float)

    def native_flux(
        self, values: Mapping[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        from robert_exoplanets import (
            AtmosphereState,
            EvaluatedCorrelatedKOpacity,
            assemble_gas_optical_depth,
            cia_optical_depth,
            gauss_legendre_disk_geometry,
            planck_radiance_wavelength,
            solve_emission,
        )
        from robert_exoplanets.rt import solve_thermal_sh4_spectrum

        state = atmospheric_state(self.common, self.scenario.name, values)
        composition = {
            name: np.full(self.pressure.n_layers, state.gas_vmr[index])
            for index, name in enumerate(state.gas_names)
        }
        atmosphere = AtmosphereState(
            pressure_grid=self.pressure,
            temperature=state.temperature_cells_k,
            temperature_edges=state.temperature_edges_k,
            composition=composition,
            mean_molecular_weight=np.full(
                self.pressure.n_layers, state.mean_molecular_weight_u
            ),
        )
        evaluated = np.empty(
            (
                len(MOLECULAR_SPECIES),
                self.pressure.n_layers,
                self.wavelength.size,
                self.g_weights.size,
            )
        )
        for index, name in enumerate(MOLECULAR_SPECIES):
            evaluated[index] = (
                self.providers[name]
                .evaluate(atmosphere, self.prepared_species[name])
                .kcoeff[0]
            )
        opacity = EvaluatedCorrelatedKOpacity(
            prepared=self.prepared,
            kcoeff=evaluated,
            unit="cm^2/molecule",
            metadata={"stage": "9"},
        )
        gas = assemble_gas_optical_depth(
            atmosphere,
            opacity,
            gravity_m_s2=float(
                self.common["derived_quantities"]["surface_gravity_m_s2"]
            ),
            gas_combination="random_overlap",
        )
        additions = [
            cia_optical_depth(
                gas,
                table,
                coefficient_interpolation="log",
                temperature_extrapolation="clip",
                spectral_extrapolation="zero",
            )
            for table in self.cia_tables.values()
        ]
        if not self.scenario.cloudy:
            result = solve_emission(
                gas,
                geometry=gauss_legendre_disk_geometry(n_mu=8),
                bottom_boundary="blackbody",
                additional_optical_depths=additions,
                multiple_scattering_backend="none",
            )
            return self.wavelength, np.pi * np.asarray(result.radiance.values)

        total_gas = np.asarray(gas.total_tau, dtype=float)
        total_gas += sum(
            np.asarray(item.tau, dtype=float)[..., None] for item in additions
        )
        cloud = power_law_cloud_tau(
            self.pressure.edges,
            self.wavelength,
            optical_depth_at_reference=state.cloud_tau_5um,
            cloud_top_pressure_bar=state.cloud_top_pressure_bar,
            extinction_slope=0.0,
        )[..., None]
        extinction = total_gas + cloud
        omega = np.divide(
            cloud * state.cloud_single_scattering_albedo,
            extinction,
            out=np.zeros_like(extinction),
            where=extinction > 0.0,
        )
        levels = np.stack(
            [
                planck_radiance_wavelength(self.wavelength, temperature)
                for temperature in state.temperature_edges_k
            ]
        )
        geometry = gauss_legendre_disk_geometry(n_mu=8)
        result = solve_thermal_sh4_spectrum(
            extinction,
            omega,
            np.zeros_like(extinction),
            levels,
            geometry.emission_angle_cosines,
            geometry.emission_angle_weights,
            self.g_weights,
            bottom_planck_radiance=levels[-1],
            delta_m=False,
            backend="numpy",
        )
        return self.wavelength, np.pi * np.asarray(result.radiance)


def _picaso_pair_name(pair: Any) -> str | None:
    values = tuple(str(value) for value in pair)
    if values == ("H2", "H2"):
        return "H2-H2"
    if len(values) == 2 and set(values) == {"H2", "He"}:
        return "H2-He"
    return None


def _patch_picaso_opacity(opacity: Any) -> None:
    original_mix = opacity.mix_my_opacities_gasesfly

    def absolute_vmr_mixer(atmosphere: Any, exclude_mol: Any = 1) -> Any:
        result = original_mix(atmosphere, exclude_mol=exclude_mol)
        total_vmr = np.sum(
            [
                atmosphere.layer["mixingratios"][name].values
                for name in atmosphere.molecules
            ],
            axis=0,
        )
        opacity.molecular_opa *= total_vmr[:, None, None]
        return result

    opacity.mix_my_opacities_gasesfly = absolute_vmr_mixer
    original_get = opacity.get_opacities

    def selected_continuum(atmosphere: Any, exclude_mol: Any = 1) -> Any:
        atmosphere.continuum_molecules = [
            pair
            for pair in atmosphere.continuum_molecules
            if _picaso_pair_name(pair) in {"H2-H2", "H2-He"}
        ]
        atmosphere.rayleigh_molecules = []
        return original_get(atmosphere, exclude_mol=exclude_mol)

    opacity.get_opacities = selected_continuum


class PicasoNativeForward(NativeForward):
    """PICASO-4 native resort-rebin SH4 path."""

    framework = "picaso"

    def __init__(self, common: Mapping[str, Any], scenario_name: str) -> None:
        super().__init__(common, scenario_name)
        from picaso import justdoit as jdi

        reference = Path(os.environ["picaso_refdata"]).expanduser().resolve()
        ck = Path(os.environ["STAGE9_PICASO_CK_DIRECTORY"]).expanduser().resolve()
        if not reference.is_dir() or not ck.is_dir():
            raise RuntimeError(
                "PICASO reference and resort-rebin directories must exist"
            )
        self.opacity = jdi.opannection(
            method="resortrebin",
            ck_db=str(ck),
            preload_gases=list(MOLECULAR_SPECIES),
            wave_range=[0.3, 12.0],
            verbose=False,
        )
        _patch_picaso_opacity(self.opacity)
        self._picaso_wavenumber = np.asarray(self.opacity.wno, dtype=float)
        self._picaso_delta_wavenumber = np.asarray(self.opacity.delta_wno, dtype=float)
        (
            self._picaso_wavelength,
            self._picaso_bin_lower_micron,
            self._picaso_bin_upper_micron,
            self._picaso_wavelength_order,
        ) = _picaso_native_wavelength_support(
            self._picaso_wavenumber,
            self._picaso_delta_wavenumber,
        )
        self.native_binning_method = "picaso_native_bin_support_overlap"

    def native_flux(
        self, values: Mapping[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        import astropy.units as u
        import pandas as pd
        from picaso import justdoit as jdi

        state = atmospheric_state(self.common, self.scenario.name, values)
        profile: dict[str, Any] = {
            "pressure": self.pressure.edges,
            "temperature": state.temperature_edges_k,
        }
        for index, name in enumerate(state.gas_names):
            profile[name] = np.full(self.pressure.edges.size, state.gas_vmr[index])
        case = jdi.inputs(calculation="browndwarf")
        case.gravity(
            gravity=float(self.common["derived_quantities"]["surface_gravity_m_s2"]),
            gravity_unit=u.m / u.s**2,
        )
        case.atmosphere(df=pd.DataFrame(profile), verbose=False)
        case.approx(
            rt_method="SH",
            stream=4,
            delta_eddington=False,
            raman="none",
            w_single_rayleigh="off",
            w_multi_rayleigh="off",
            psingle_rayleigh="off",
        )
        if self.scenario.cloudy:
            tau = power_law_cloud_tau(
                self.pressure.edges,
                1.0e4 / self._picaso_wavenumber,
                optical_depth_at_reference=state.cloud_tau_5um,
                cloud_top_pressure_bar=state.cloud_top_pressure_bar,
                extinction_slope=0.0,
            )
            rows = [
                (
                    self.pressure.centers[layer],
                    self._picaso_wavenumber[wave],
                    tau[layer, wave],
                    state.cloud_single_scattering_albedo,
                    0.0,
                )
                for layer in range(self.pressure.n_layers)
                for wave in range(self._picaso_wavenumber.size)
            ]
            case.clouds(
                df=pd.DataFrame(
                    rows, columns=("pressure", "wavenumber", "opd", "w0", "g0")
                )
            )
        result = case.spectrum(self.opacity, calculation="thermal", full_output=False)
        result_wavenumber = np.asarray(result["wavenumber"], dtype=float)
        if result_wavenumber.shape != self._picaso_wavenumber.shape or not np.allclose(
            result_wavenumber, self._picaso_wavenumber, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError(
                "PICASO output grid differs from its native opacity-bin grid"
            )
        native_flux = np.asarray(result["thermal"], dtype=float) * 0.1
        return (
            self._picaso_wavelength,
            native_flux[self._picaso_wavelength_order],
        )

    def eclipse_depth(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        """Project PICASO's native bins without interpolating their centres."""

        wavelength, native_flux = self.native_flux(values)
        binned = _native_bin_overlap_mean(
            self._picaso_bin_lower_micron,
            self._picaso_bin_upper_micron,
            native_flux,
            self.r100_edges,
        )
        self.last_native_wavelength = np.asarray(wavelength, dtype=float)
        self.last_native_flux = np.asarray(native_flux, dtype=float)
        self.last_native_bin_lower_micron = self._picaso_bin_lower_micron.copy()
        self.last_native_bin_upper_micron = self._picaso_bin_upper_micron.copy()
        return np.asarray(binned / self.stellar_r100 * self.area_ratio, dtype=float)


class PetitRadtransNativeForward(NativeForward):
    """petitRADTRANS 3.3.3 native correlated-k/CIA Feautrier path."""

    framework = "petitradtrans"

    def __init__(self, common: Mapping[str, Any], scenario_name: str) -> None:
        super().__init__(common, scenario_name)
        from petitRADTRANS.radtrans import Radtrans

        root = Path(os.environ["STAGE9_PRT_INPUT_DATA"]).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(
                "STAGE9_PRT_INPUT_DATA must name a staged pRT input_data tree"
            )
        nodes, weights = np.polynomial.legendre.leggauss(8)
        angle_grid = np.vstack((0.5 * (nodes + 1.0), 0.5 * weights))
        self.atmosphere = Radtrans(
            pressures=self.pressure.centers,
            wavelength_boundaries=np.array([0.3, 12.1]),
            line_species=list(PRT_LINE_SPECIES.values()),
            gas_continuum_contributors=list(PRT_CIA_SPECIES.values()),
            rayleigh_species=[],
            cloud_species=[],
            scattering_in_emission=self.scenario.cloud == "grey_isotropic_scattering",
            anisotropic_cloud_scattering=False,
            emission_angle_grid=angle_grid,
            path_input_data=str(root),
        )

    def native_flux(
        self, values: Mapping[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        state = atmospheric_state(self.common, self.scenario.name, values)
        fractions = state.gas_vmr * state.gas_mass_u / state.mean_molecular_weight_u
        mass = {
            line: np.full(
                self.pressure.n_layers, fractions[state.gas_names.index(name)]
            )
            for name, line in PRT_LINE_SPECIES.items()
        }
        for name in ("H2", "He"):
            mass[name] = np.full(
                self.pressure.n_layers, fractions[state.gas_names.index(name)]
            )
        pressure_thickness = np.diff(self.pressure.edges) * 1.0e6
        gravity_cgs = (
            float(self.common["derived_quantities"]["surface_gravity_m_s2"]) * 100.0
        )
        vertical_tau = power_law_cloud_tau(
            self.pressure.edges,
            self.r100_centers,
            optical_depth_at_reference=state.cloud_tau_5um,
            cloud_top_pressure_bar=state.cloud_top_pressure_bar,
            extinction_slope=0.0,
        )
        mass_opacity = vertical_tau * gravity_cgs / pressure_thickness[:, None]

        def total_cloud(
            wavelength: NDArray[np.float64], pressure: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            del pressure
            return np.stack(
                [np.interp(wavelength, self.r100_centers, row) for row in mass_opacity],
                axis=1,
            )

        def absorption(
            wavelength: NDArray[np.float64], pressure: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return total_cloud(wavelength, pressure) * (
                1.0 - state.cloud_single_scattering_albedo
            )

        def scattering(
            wavelength: NDArray[np.float64], pressure: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return (
                total_cloud(wavelength, pressure) * state.cloud_single_scattering_albedo
            )

        result = self.atmosphere.calculate_flux(
            temperatures=state.temperature_cells_k,
            mass_fractions=mass,
            mean_molar_masses=np.full(
                self.pressure.n_layers, state.mean_molecular_weight_u
            ),
            reference_gravity=gravity_cgs,
            additional_absorption_opacities_function=absorption
            if self.scenario.cloudy
            else None,
            additional_scattering_opacities_function=(
                scattering
                if self.scenario.cloud == "grey_isotropic_scattering"
                else None
            ),
            frequencies_to_wavelengths=True,
            return_contribution=False,
        )
        wavelength = np.asarray(result[0], dtype=float) * 1.0e4
        order = np.argsort(wavelength)
        return wavelength[order], np.asarray(result[1], dtype=float)[order] * 0.1


def build_native_forward(
    framework: str, common: Mapping[str, Any], scenario_name: str
) -> NativeForward:
    """Construct one supported persistent native adapter."""

    adapters = {
        "robert": RobertNativeForward,
        "picaso": PicasoNativeForward,
        "petitradtrans": PetitRadtransNativeForward,
    }
    try:
        adapter = adapters[framework]
    except KeyError as exc:
        raise ValueError(f"unsupported Stage-9 framework: {framework}") from exc
    return adapter(common, scenario_name)


def load_common_contract(path: str | Path) -> dict[str, Any]:
    """Load a frozen common contract without silently accepting NaN."""

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


__all__ = [
    "NativeForward",
    "atmospheric_state",
    "build_native_forward",
    "load_common_contract",
    "truth_parameters",
]
