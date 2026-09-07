"""Real-pair, float32 homogeneous-sphere Mie optics for JAX Metal.

This is an independent translation of ROBERT's reference Mie recurrence.  It
uses ``(real, imaginary)`` pairs instead of JAX complex arrays, preserving the
full Lorenz--Mie calculation and scalar Legendre moments through degree four.
Array sizes are explicit static inputs; callers should reuse a compiled shape.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robert_exoplanets.core import RobertConfigError, RobertValidationError

_KERNELS: OrderedDict[tuple[int, int], Any] = OrderedDict()


def _runtime():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RobertConfigError(
            "the Metal Mie backend requires robert-exoplanets[jax]"
        ) from exc
    return jax, jnp


def _validate_order(maximum_order: int) -> int:
    value = int(maximum_order)
    if value < 2:
        raise RobertValidationError("maximum_order must be at least two")
    return value


def _validate_downward_order(maximum_downward_order: int, maximum_order: int) -> int:
    value = int(maximum_downward_order)
    if value < maximum_order:
        raise RobertValidationError(
            "maximum_downward_order must be at least maximum_order"
        )
    return value


def _leggauss(order: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    mu, weight = np.polynomial.legendre.leggauss(int(order))
    return np.asarray(mu, dtype=np.float32), np.asarray(weight, dtype=np.float32)


def mie_efficiencies_and_moments_device(
    size_parameter: Any,
    real_index: Any,
    imaginary_index: Any,
    quadrature_mu: Any,
    quadrature_weight: Any,
    *,
    maximum_order: int,
    maximum_downward_order: int,
) -> tuple[Any, Any, Any, Any]:
    """Device-native exact Mie calculation with no host transfer.

    Inputs and outputs are JAX arrays.  ``maximum_order`` and the quadrature
    shape are static compilation choices.  This function intentionally has no
    NumPy conversions or Python value validation so it can be called inside an
    outer ``jax.jit`` encompassing the complete forward model.
    """
    nmax = _validate_order(maximum_order)
    ndown = _validate_downward_order(maximum_downward_order, nmax)
    jax, jnp = _runtime()
    return _kernel(jax, jnp, nmax, ndown)(
        jnp.asarray(size_parameter, dtype=jnp.float32),
        jnp.asarray(real_index, dtype=jnp.float32),
        jnp.asarray(imaginary_index, dtype=jnp.float32),
        jnp.asarray(quadrature_mu, dtype=jnp.float32),
        jnp.asarray(quadrature_weight, dtype=jnp.float32),
    )


def lognormal_mie_optics_device(
    wavelength_micron: Any,
    real_index: Any,
    imaginary_index: Any,
    effective_radius_micron: Any,
    geometric_stddev: Any,
    particle_density_kg_m3: Any,
    size_nodes: Any,
    size_weights: Any,
    angular_mu: Any,
    angular_weight: Any,
    *,
    maximum_order: int,
    maximum_downward_order: int,
) -> dict[str, Any]:
    """Device-native exact lognormal Mie optics with real-pair arithmetic.

    ``size_nodes``/``size_weights`` are the fixed Gauss--Legendre rule used by
    the CPU reference (or a single node of zero with unit weight for a
    monodisperse distribution).  All interpolation is deliberately outside
    this primitive; supplied refractive-index arrays are already on the model
    wavelength grid.
    """
    jax, jnp = _runtime()
    wave = jnp.asarray(wavelength_micron, dtype=jnp.float32)
    nr = jnp.asarray(real_index, dtype=jnp.float32)
    ni = jnp.asarray(imaginary_index, dtype=jnp.float32)
    nodes = jnp.asarray(size_nodes, dtype=jnp.float32)
    weights = jnp.asarray(size_weights, dtype=jnp.float32)
    radius = jnp.asarray(effective_radius_micron, dtype=jnp.float32)
    width = jnp.asarray(geometric_stddev, dtype=jnp.float32)
    density = jnp.asarray(particle_density_kg_m3, dtype=jnp.float32)
    coordinate = jnp.float32(6.0) * nodes
    raw_weight = (
        jnp.float32(6.0)
        * weights
        * jnp.exp(-jnp.float32(0.5) * coordinate**2)
        / jnp.sqrt(jnp.float32(2.0 * np.pi))
    )
    log_width = jnp.log(width)
    radii_lognormal = radius * jnp.exp(
        -jnp.float32(2.5) * log_width**2 + log_width * coordinate
    )
    # Static one-node rules are accepted for width=1; select analytically so
    # this remains traceable for dynamic retrieval parameters.
    is_mono = width == jnp.float32(1.0)
    radii = jnp.where(is_mono, jnp.full_like(nodes, radius), radii_lognormal)
    number_weight = jnp.where(is_mono, jnp.ones_like(weights), raw_weight)
    number_weight = number_weight / jnp.sum(number_weight)
    x = (jnp.float32(2.0 * np.pi) * radii[:, None] / wave[None, :]).reshape(-1)
    qext, qsca, _, moments = mie_efficiencies_and_moments_device(
        x,
        jnp.tile(nr, radii.size),
        jnp.tile(ni, radii.size),
        angular_mu,
        angular_weight,
        maximum_order=maximum_order,
        maximum_downward_order=maximum_downward_order,
    )
    qext = qext.reshape(radii.size, wave.size)
    qsca = qsca.reshape(radii.size, wave.size)
    moments = moments.reshape(radii.size, wave.size, 5)
    radii_m = radii * jnp.float32(1.0e-6)
    area_weight = number_weight * jnp.float32(3.141592653589793) * radii_m**2
    mean_mass = jnp.sum(
        number_weight
        * (
            jnp.float32(4.0 / 3.0)
            * jnp.float32(3.141592653589793)
            * density
            * radii_m**3
        )
    )
    extinction = jnp.sum(area_weight[:, None] * qext, axis=0)
    scattering = jnp.sum(area_weight[:, None] * qsca, axis=0)
    scatter_moments = jnp.sum(
        area_weight[:, None, None] * qsca[:, :, None] * moments, axis=0
    ).T
    mass_extinction = extinction / mean_mass
    mass_scattering = scattering / mean_mass
    phase = jnp.where(
        scattering[None, :] > 0.0,
        scatter_moments / scattering[None, :],
        jnp.zeros_like(scatter_moments),
    )
    phase = phase.at[0].set(jnp.where(scattering > 0.0, phase[0], 1.0))
    valid = (
        jnp.all(jnp.isfinite(wave))
        & jnp.all(wave > 0.0)
        & jnp.all(jnp.isfinite(nr))
        & jnp.all(nr > 0.0)
        & jnp.all(jnp.isfinite(ni))
        & jnp.all(ni >= 0.0)
        & jnp.isfinite(radius)
        & (radius > 0.0)
        & jnp.isfinite(width)
        & (width >= 1.0)
        & jnp.isfinite(density)
        & (density > 0.0)
        & jnp.all(jnp.isfinite(mass_extinction))
        & jnp.all(jnp.isfinite(mass_scattering))
        & jnp.all(jnp.isfinite(phase))
    )
    invalid = jnp.float32(jnp.nan)
    mass_extinction = jnp.where(valid, mass_extinction, invalid)
    mass_scattering = jnp.where(valid, mass_scattering, invalid)
    phase = jnp.where(valid, phase, invalid)
    return {
        "mass_extinction_m2_kg": mass_extinction,
        "mass_scattering_m2_kg": mass_scattering,
        "single_scattering_albedo": jnp.clip(
            jnp.where(mass_extinction > 0.0, mass_scattering / mass_extinction, 0.0),
            0.0,
            1.0,
        ),
        "asymmetry_factor": jnp.clip(phase[1] / jnp.float32(3.0), -1.0, 1.0),
        "phase_function_moments": phase,
    }


def mie_efficiencies_and_moments(
    size_parameter: ArrayLike,
    real_index: ArrayLike | float,
    imaginary_index: ArrayLike | float,
    *,
    maximum_order: int,
    maximum_downward_order: int | None = None,
    platform: str | None = None,
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    """Evaluate exact Mie ``Qext``, ``Qsca``, ``g`` and five moments.

    ``maximum_order`` is the fixed recurrence capacity, and must exceed the
    Bohren--Huffman order required by every input.  Values above the needed
    order are masked, so a single static shape supports a family of radii.
    The returned values are copied to host for the explicit research API.
    """
    nmax = _validate_order(maximum_order)
    x = np.asarray(size_parameter, dtype=np.float32)
    nr = np.broadcast_to(np.asarray(real_index, dtype=np.float32), x.shape)
    ni = np.broadcast_to(np.asarray(imaginary_index, dtype=np.float32), x.shape)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise RobertValidationError(
            "size_parameter must be a finite positive one-dimensional array"
        )
    if (
        not np.all(np.isfinite(nr))
        or not np.all(np.isfinite(ni))
        or np.any(nr <= 0.0)
        or np.any(ni < 0.0)
    ):
        raise RobertValidationError(
            "refractive indices require finite n > 0 and k >= 0"
        )
    required = np.floor(x + 4.05 * np.cbrt(x) + 2.0).astype(int)
    if np.any(required > nmax):
        raise RobertValidationError(
            f"maximum_order={nmax} is too small for size parameter requiring {int(required.max())} terms"
        )
    required_downward = np.maximum(
        required + 15, np.floor(np.abs((nr + 1j * ni) * x)).astype(int) + 15
    )
    ndown = (
        max(nmax, int(required_downward.max()))
        if maximum_downward_order is None
        else int(maximum_downward_order)
    )
    _validate_downward_order(ndown, nmax)
    if np.any(required_downward > ndown):
        raise RobertValidationError(
            "maximum_downward_order is too small for the supplied refractive-index and size-parameter bounds"
        )
    jax, jnp = _runtime()
    device = _device(jax, platform)
    # nmax+7 quadrature nodes exactly integrate the intensity polynomial and
    # the five requested Legendre moments for the active recurrence terms.
    mu, weight = _leggauss(max(16, nmax + 7))
    result = mie_efficiencies_and_moments_device(
        jax.device_put(jnp.asarray(x), device),
        jax.device_put(jnp.asarray(nr), device),
        jax.device_put(jnp.asarray(ni), device),
        jax.device_put(jnp.asarray(mu), device),
        jax.device_put(jnp.asarray(weight), device),
        maximum_order=nmax,
        maximum_downward_order=ndown,
    )
    result = tuple(item.block_until_ready() for item in result)
    return tuple(np.asarray(jax.device_get(item), dtype=np.float32) for item in result)  # type: ignore[return-value]


def lognormal_mie_optics(
    wavelength_micron: ArrayLike,
    real_index: ArrayLike,
    imaginary_index: ArrayLike,
    *,
    effective_radius_micron: float,
    geometric_stddev: float,
    particle_density_kg_m3: float,
    quadrature_points: int = 24,
    maximum_order: int,
    maximum_downward_order: int | None = None,
    platform: str | None = None,
) -> dict[str, NDArray[np.float32]]:
    """Return exact lognormal Mie mass optics, using the CPU reference convention.

    Refractive indices must already have been interpolated onto ``wavelength``.
    This intentionally returns plain arrays, keeping it separate from ROBERT's
    CPU domain containers while the Metal backend remains experimental.
    """
    wave = np.asarray(wavelength_micron, dtype=np.float32)
    nr = np.asarray(real_index, dtype=np.float32)
    ni = np.asarray(imaginary_index, dtype=np.float32)
    radius, width, density = (
        float(effective_radius_micron),
        float(geometric_stddev),
        float(particle_density_kg_m3),
    )
    points = int(quadrature_points)
    if (
        wave.ndim != 1
        or wave.size < 1
        or nr.shape != wave.shape
        or ni.shape != wave.shape
        or np.any(wave <= 0.0)
    ):
        raise RobertValidationError(
            "wavelength and refractive-index arrays must be matching positive one-dimensional arrays"
        )
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or not np.isfinite(density)
        or density <= 0.0
        or not np.isfinite(width)
        or width < 1.0
        or points < 1
    ):
        raise RobertValidationError("invalid lognormal Mie parameters")
    if width == 1.0:
        radii, number_weight = (
            np.array([radius], np.float32),
            np.array([1.0], np.float32),
        )
    else:
        node, weight = _leggauss(points)
        coordinate = np.float32(6.0) * node
        number_weight = (
            np.float32(6.0)
            * weight
            * np.exp(-np.float32(0.5) * coordinate**2)
            / np.sqrt(np.float32(2.0 * np.pi))
        )
        number_weight /= np.sum(number_weight)
        mean_radius = np.float32(radius) * np.exp(
            -np.float32(2.5) * np.log(np.float32(width)) ** 2
        )
        radii = mean_radius * np.exp(np.log(np.float32(width)) * coordinate)
    x = (np.float32(2.0 * np.pi) * radii[:, None] / wave[None, :]).reshape(-1)
    qext, qsca, _, moments = mie_efficiencies_and_moments(
        x,
        np.tile(nr, radii.size),
        np.tile(ni, radii.size),
        maximum_order=maximum_order,
        maximum_downward_order=maximum_downward_order,
        platform=platform,
    )
    qext, qsca = (
        qext.reshape(radii.size, wave.size),
        qsca.reshape(radii.size, wave.size),
    )
    moments = moments.reshape(radii.size, wave.size, 5)
    radii_m = radii * np.float32(1.0e-6)
    area_weight = number_weight * np.float32(np.pi) * radii_m**2
    mean_mass = np.sum(
        number_weight
        * (np.float32(4.0 / 3.0) * np.float32(np.pi) * np.float32(density) * radii_m**3)
    )
    extinction = np.sum(area_weight[:, None] * qext, axis=0)
    scattering = np.sum(area_weight[:, None] * qsca, axis=0)
    scatter_moments = np.sum(
        area_weight[:, None, None] * qsca[:, :, None] * moments, axis=0
    ).T
    mass_extinction = extinction / mean_mass
    mass_scattering = scattering / mean_mass
    phase = np.divide(
        scatter_moments,
        scattering[None, :],
        out=np.zeros_like(scatter_moments),
        where=scattering[None, :] > 0.0,
    )
    phase[0, scattering <= 0.0] = 1.0
    return {
        "mass_extinction_m2_kg": mass_extinction,
        "mass_scattering_m2_kg": mass_scattering,
        "single_scattering_albedo": np.clip(
            np.divide(
                mass_scattering,
                mass_extinction,
                out=np.zeros_like(mass_extinction),
                where=mass_extinction > 0.0,
            ),
            0.0,
            1.0,
        ),
        "asymmetry_factor": np.clip(phase[1] / np.float32(3.0), -1.0, 1.0),
        "phase_function_moments": phase,
    }


def cloud_extinction_tau(
    layer_pressure_thickness_pa: ArrayLike,
    gravity_m_s2: float,
    condensate_mass_fraction: ArrayLike | float,
    mass_extinction_m2_kg: ArrayLike,
) -> NDArray[np.float32]:
    """Construct ``tau=kappa*q*delta_pressure/gravity`` for Mie clouds."""
    thickness = np.asarray(layer_pressure_thickness_pa, dtype=np.float32)
    opacity = np.asarray(mass_extinction_m2_kg, dtype=np.float32)
    fraction = np.asarray(condensate_mass_fraction, dtype=np.float32)
    gravity = float(gravity_m_s2)
    if (
        thickness.ndim != 1
        or opacity.ndim != 1
        or gravity <= 0.0
        or not np.all(np.isfinite(thickness))
        or np.any(thickness < 0.0)
    ):
        raise RobertValidationError("invalid hydrostatic cloud inputs")
    if fraction.ndim == 0:
        fraction = np.full(thickness.shape, fraction, dtype=np.float32)
    if (
        fraction.shape != thickness.shape
        or np.any(fraction < 0.0)
        or np.any(fraction > 1.0)
    ):
        raise RobertValidationError(
            "condensate_mass_fraction must lie in [0, 1] per layer"
        )
    return (
        thickness[:, None] / np.float32(gravity) * fraction[:, None] * opacity[None, :]
    )


def cloud_extinction_tau_device(
    layer_pressure_thickness_pa: Any,
    gravity_m_s2: Any,
    condensate_mass_fraction: Any,
    mass_extinction_m2_kg: Any,
) -> Any:
    """Device-native hydrostatic Mie-cloud optical-depth construction."""
    _, jnp = _runtime()
    thickness = jnp.asarray(layer_pressure_thickness_pa, dtype=jnp.float32)
    fraction = jnp.asarray(condensate_mass_fraction, dtype=jnp.float32)
    opacity = jnp.asarray(mass_extinction_m2_kg, dtype=jnp.float32)
    gravity = jnp.asarray(gravity_m_s2, dtype=jnp.float32)
    tau = thickness[:, None] / gravity * fraction[:, None] * opacity[None, :]
    valid = (
        jnp.all(jnp.isfinite(thickness))
        & jnp.all(thickness > 0.0)
        & jnp.isfinite(gravity)
        & (gravity > 0.0)
        & jnp.all(jnp.isfinite(fraction))
        & jnp.all(fraction >= 0.0)
        & jnp.all(fraction <= 1.0)
        & jnp.all(jnp.isfinite(opacity))
        & jnp.all(opacity >= 0.0)
    )
    return jnp.where(valid, tau, jnp.nan)


def _device(jax: Any, platform: str | None):
    requested = (
        None
        if platform is None
        else ("gpu" if platform.lower() in {"metal", "gpu"} else platform.lower())
    )
    try:
        devices = jax.devices(requested) if requested else jax.devices()
    except RuntimeError as exc:
        raise RobertConfigError(
            f"requested JAX platform {platform!r} is unavailable"
        ) from exc
    if not devices:
        raise RobertConfigError(
            f"requested JAX platform {platform!r} has no visible devices"
        )
    return devices[0]


def _kernel(jax: Any, jnp: Any, nmax: int, ndown: int):
    cached = _KERNELS.get((nmax, ndown))
    if cached is not None:
        _KERNELS.move_to_end((nmax, ndown))
        return cached
    lax = jax.lax

    def cmul(ar, ai, br, bi):
        return ar * br - ai * bi, ar * bi + ai * br

    def cdiv(ar, ai, br, bi):
        denom = br * br + bi * bi
        return (ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom

    def cabs2(ar, ai):
        return ar * ar + ai * ai

    def one(x, mr, mi, mu, weight):
        # Static capacity, but each vmap lane starts at the same n_down as the
        # CPU solver: max(n_stop + 15, int(abs(m*x)) + 15).
        zre, zim = mr * x, mi * x
        nstop_raw = jnp.floor(x + 4.05 * x ** (1.0 / 3.0) + 2.0)
        ndown_lane = jnp.maximum(
            nstop_raw + 15.0, jnp.floor(jnp.sqrt(zre * zre + zim * zim)) + 15.0
        )

        # Store downward D_n by a second fixed recurrence, indexed from high
        # to low.  We retain it in a static vector then reverse it.
        def down_store(i, carry):
            dr, di, arr_r, arr_i = carry
            n = jnp.float32(ndown - i)
            nr, ni = cdiv(n, 0.0, zre, zim)
            ir, ii = cdiv(1.0, 0.0, dr + nr, di + ni)
            newr, newi = nr - ir, ni - ii
            active = n <= ndown_lane
            next_r = jnp.where(active, newr, dr)
            next_i = jnp.where(active, newi, di)
            index = (n - 1).astype(jnp.int32)
            return (
                next_r,
                next_i,
                arr_r.at[index].set(jnp.where(active, newr, 0.0)),
                arr_i.at[index].set(jnp.where(active, newi, 0.0)),
            )

        zeros = jnp.zeros(ndown, dtype=jnp.float32)
        _, _, drs, dis = lax.fori_loop(0, ndown, down_store, (0.0, 0.0, zeros, zeros))
        nstop = jnp.minimum(jnp.float32(nmax), nstop_raw)
        psi0, psi1 = jnp.sin(x), jnp.sin(x) / x - jnp.cos(x)
        chi0, chi1 = jnp.cos(x), jnp.cos(x) / x + jnp.sin(x)
        ar = jnp.zeros(nmax, dtype=jnp.float32)
        ai = jnp.zeros(nmax, dtype=jnp.float32)
        br = jnp.zeros(nmax, dtype=jnp.float32)
        bi = jnp.zeros(nmax, dtype=jnp.float32)

        def upward(i, carry):
            ps0, ps1, ch0, ch1, aa_r, aa_i, bb_r, bb_i = carry
            n = jnp.float32(i + 1)
            d_r, d_i = drs[i + 1], dis[i + 1]
            dm_r, dm_i = cdiv(d_r, d_i, mr, mi)
            af_r, af_i = dm_r + n / x, dm_i
            bm_r, bm_i = cmul(mr, mi, d_r, d_i)
            bm_r += n / x
            xi0r, xi0i, xi1r, xi1i = ps0, -ch0, ps1, -ch1
            anr, ani = cdiv(
                af_r * ps1 - ps0,
                af_i * ps1,
                af_r * xi1r - af_i * xi1i - xi0r,
                af_r * xi1i + af_i * xi1r - xi0i,
            )
            bnr, bni = cdiv(
                bm_r * ps1 - ps0,
                bm_i * ps1,
                bm_r * xi1r - bm_i * xi1i - xi0r,
                bm_r * xi1i + bm_i * xi1r - xi0i,
            )
            active = n <= nstop
            aa_r = aa_r.at[i].set(jnp.where(active, anr, 0.0))
            aa_i = aa_i.at[i].set(jnp.where(active, ani, 0.0))
            bb_r = bb_r.at[i].set(jnp.where(active, bnr, 0.0))
            bb_i = bb_i.at[i].set(jnp.where(active, bni, 0.0))
            ps2 = (2.0 * n + 1.0) * ps1 / x - ps0
            ch2 = (2.0 * n + 1.0) * ch1 / x - ch0
            return ps1, ps2, ch1, ch2, aa_r, aa_i, bb_r, bb_i

        _, _, _, _, ar, ai, br, bi = lax.fori_loop(
            0, nmax, upward, (psi0, psi1, chi0, chi1, ar, ai, br, bi)
        )
        order = jnp.arange(1, nmax + 1, dtype=jnp.float32)
        w = 2.0 * order + 1.0
        qext = 2.0 * jnp.sum(w * (ar + br)) / x**2
        qsca = 2.0 * jnp.sum(w * (cabs2(ar, ai) + cabs2(br, bi))) / x**2

        # Exact angular quadrature phase moments, expressed in real pairs.
        def angle_eval(m):
            pi0, pi1 = jnp.float32(0.0), jnp.float32(1.0)
            s1r = s1i = s2r = s2i = jnp.float32(0.0)

            def angular(i, state):
                p0, p1, u1r, u1i, u2r, u2i = state
                n = jnp.float32(i + 1)
                tau = n * m * p1 - (n + 1.0) * p0
                fac = (2.0 * n + 1.0) / (n * (n + 1.0))
                return (
                    p1,
                    ((2.0 * n + 1.0) * m * p1 - (n + 1.0) * p0) / n,
                    u1r + fac * (ar[i] * p1 + br[i] * tau),
                    u1i + fac * (ai[i] * p1 + bi[i] * tau),
                    u2r + fac * (ar[i] * tau + br[i] * p1),
                    u2i + fac * (ai[i] * tau + bi[i] * p1),
                )

            *_, u1r, u1i, u2r, u2i = lax.fori_loop(
                0, nmax, angular, (pi0, pi1, s1r, s1i, s2r, s2i)
            )
            return 0.5 * (u1r * u1r + u1i * u1i + u2r * u2r + u2i * u2i)

        intensity = jax.vmap(angle_eval)(mu)
        norm = jnp.sum(weight * intensity)
        p0 = jnp.ones_like(mu)
        p1 = mu
        p2 = 0.5 * (3 * mu * mu - 1)
        p3 = (5 * mu * p2 - 2 * p1) / 3
        p4 = (7 * mu * p3 - 3 * p2) / 4
        basis = jnp.stack((p0, p1, p2, p3, p4))
        moments = (
            (2 * jnp.arange(5, dtype=jnp.float32) + 1)
            * jnp.sum(basis * (weight * intensity), axis=1)
            / norm
        )
        moments = moments.at[0].set(1.0)
        # Analytic Rayleigh branch matches the CPU reference below x=1e-3.
        pr, pi = cdiv(
            mr * mr - mi * mi - 1.0, 2 * mr * mi, mr * mr - mi * mi + 2.0, 2 * mr * mi
        )
        ray_sca = (8.0 / 3.0) * x**4 * cabs2(pr, pi)
        ray_abs = jnp.maximum(0.0, 4 * x * pi)
        ray_mom = jnp.array((1.0, 0.0, 0.5, 0.0, 0.0), dtype=jnp.float32)
        tiny = x < 1e-3
        final_qext = jnp.where(tiny, ray_abs + ray_sca, jnp.maximum(qext, 0.0))
        final_qsca = jnp.where(tiny, ray_sca, jnp.clip(qsca, 0.0, qext * (1 + 1e-5)))
        final_g = jnp.where(tiny, 0.0, moments[1] / 3.0)
        final_moments = jnp.where(tiny, ray_mom, moments)
        valid = (
            jnp.isfinite(x)
            & jnp.isfinite(mr)
            & jnp.isfinite(mi)
            & (x > 0.0)
            & (mr > 0.0)
            & (mi >= 0.0)
            & (nstop_raw <= jnp.float32(nmax))
            & (ndown_lane <= jnp.float32(ndown))
            & jnp.isfinite(final_qext)
            & jnp.isfinite(final_qsca)
            & jnp.all(jnp.isfinite(final_moments))
        )
        invalid = jnp.float32(jnp.nan)
        return (
            jnp.where(valid, final_qext, invalid),
            jnp.where(valid, final_qsca, invalid),
            jnp.where(valid, final_g, invalid),
            jnp.where(valid, final_moments, invalid),
        )

    compiled = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None, None)))
    _KERNELS[(nmax, ndown)] = compiled
    if len(_KERNELS) > 8:
        _KERNELS.popitem(last=False)
    return compiled


__all__ = [
    "cloud_extinction_tau",
    "cloud_extinction_tau_device",
    "lognormal_mie_optics",
    "lognormal_mie_optics_device",
    "mie_efficiencies_and_moments",
    "mie_efficiencies_and_moments_device",
]
