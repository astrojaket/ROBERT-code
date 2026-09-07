"""JAX implementation of ROBERT's P3/SH4 spectrum-only thermal solver.

This is isolated from :mod:`robert_exoplanets.rt.sh4`.  In particular, it
uses explicit pivoted band arithmetic rather than ``jnp.linalg.solve`` because
the latter lowers to an unsupported triangular solve on JAX Metal.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from robert_exoplanets.core import RobertValidationError

from .resources import MetalResourcePolicy

_KERNELS: OrderedDict[int, Any] = OrderedDict()


def metal_solve_thermal_sh4_spectrum(
    extinction_tau: Any,
    single_scattering_albedo: Any,
    asymmetry_factor: Any,
    level_planck_radiance: Any,
    emission_angle_cosines: Any,
    emission_angle_weights: Any,
    g_weights: Any,
    *,
    bottom_planck_radiance: Any,
    phase_function_moments: Any | None = None,
    delta_m_forward_fraction: Any | None = None,
    delta_m: bool = True,
    source_quadrature_order: int = 6,
):
    """Return a device-resident SH4 radiance vector using float32 or float64.

    The supplied dtype is retained.  Thus CPU JAX x64 is suitable for
    algorithmic parity, while Metal callers pass float32 device arrays.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise RobertValidationError("Metal SH4 requires JAX") from exc
    tau = jnp.asarray(extinction_tau)
    omega = jnp.asarray(single_scattering_albedo, dtype=tau.dtype)
    asymmetry = jnp.asarray(asymmetry_factor, dtype=tau.dtype)
    planck = jnp.asarray(level_planck_radiance, dtype=tau.dtype)
    bottom = jnp.asarray(bottom_planck_radiance, dtype=tau.dtype)
    mu = jnp.asarray(emission_angle_cosines, dtype=tau.dtype)
    aw = jnp.asarray(emission_angle_weights, dtype=tau.dtype)
    aw = aw / jnp.sum(aw)
    gw = jnp.asarray(g_weights, dtype=tau.dtype)
    gw = gw / jnp.sum(gw)
    if tau.ndim != 3 or omega.shape != tau.shape or asymmetry.shape != tau.shape:
        raise RobertValidationError(
            "SH4 optical arrays must share layer x spectral x g shape"
        )
    if planck.shape != (tau.shape[0] + 1, tau.shape[1]) or bottom.shape != (
        tau.shape[1],
    ):
        raise RobertValidationError("SH4 Planck array shapes are inconsistent")
    if mu.ndim != 1 or aw.shape != mu.shape or gw.shape != (tau.shape[2],):
        raise RobertValidationError("SH4 quadrature weights have incompatible shape")
    if source_quadrature_order < 2:
        raise RobertValidationError("source_quadrature_order must be at least two")
    policy = MetalResourcePolicy.laptop_safe()
    policy.require_safe(
        "SH4 graph",
        policy.estimate_sh4_bytes(
            layers=tau.shape[0],
            spectral=tau.shape[1],
            g_ordinates=tau.shape[2],
        ),
    )
    if phase_function_moments is None:
        moments = jnp.stack(
            (jnp.ones_like(tau), 3 * asymmetry, 5 * asymmetry**2, 7 * asymmetry**3)
        )
        forward = asymmetry**4
    else:
        moments = jnp.asarray(phase_function_moments, dtype=tau.dtype)
        if (
            moments.shape != (4,) + tau.shape
            or delta_m_forward_fraction is None
            and delta_m
        ):
            raise RobertValidationError(
                "supplied SH4 moments require explicit delta-M forward fraction"
            )
        forward = (
            jnp.asarray(delta_m_forward_fraction, dtype=tau.dtype)
            if delta_m
            else jnp.zeros_like(tau)
        )
    if delta_m:
        denominator = 1 - omega * forward
        tau = denominator * tau
        omega = jnp.where(denominator > 0, omega * (1 - forward) / denominator, 0)
        degree = jnp.arange(4, dtype=tau.dtype).reshape((4, 1, 1, 1))
        moments = (moments - (2 * degree + 1) * forward) / (1 - forward)
    nodes, weights = np.polynomial.legendre.leggauss(source_quadrature_order)
    kernel = _KERNELS.get(source_quadrature_order)
    if kernel is None:
        kernel = _kernel(jax, jnp, source_quadrature_order)
        _KERNELS[source_quadrature_order] = kernel
        if len(_KERNELS) > 4:
            _KERNELS.popitem(last=False)
    else:
        _KERNELS.move_to_end(source_quadrature_order)
    return kernel(
        tau,
        omega,
        moments,
        planck,
        bottom,
        mu,
        aw,
        gw,
        jnp.asarray((nodes + 1) / 2, dtype=tau.dtype),
        jnp.asarray(weights / 2, dtype=tau.dtype),
    )


def _kernel(jax, jnp, quadrature_order):
    lax = jax.lax

    def half(values):
        pi = jnp.pi
        return jnp.stack(
            (
                pi
                * (
                    values[..., 0, :] - 2 * values[..., 1, :] + 1.25 * values[..., 2, :]
                ),
                pi
                * (
                    -0.25 * values[..., 0, :]
                    + 1.25 * values[..., 2, :]
                    - 2 * values[..., 3, :]
                ),
                pi
                * (
                    values[..., 0, :] + 2 * values[..., 1, :] + 1.25 * values[..., 2, :]
                ),
                pi
                * (
                    -0.25 * values[..., 0, :]
                    + 1.25 * values[..., 2, :]
                    + 2 * values[..., 3, :]
                ),
            ),
            axis=-2,
        )

    def modes(a, layer_tau, fraction):
        beta = a[0] * a[1] + a[2] * a[3] / 9 + 4 * a[0] * a[3] / 9
        gamma = a[0] * a[1] * a[2] * a[3] / 9
        root = jnp.sqrt(jnp.maximum(beta * beta - 4 * gamma, 0))
        lam = jnp.stack((jnp.sqrt(0.5 * (beta + root)), jnp.sqrt(0.5 * (beta - root))))
        conservative = a[0] == 0
        cols = []
        for pair in range(2):
            safe = jnp.where(conservative & (pair == 1), 1, lam[pair])
            r = -a[0] / safe
            q = 0.5 * (a[0] * a[1] / safe**2 - 1)
            s = -1.5 / a[3] * (a[0] * a[1] / safe - safe)
            p = jnp.stack((jnp.ones_like(r), r, q, s), axis=-1)
            n = p * jnp.array([1, -1, 1, -1], dtype=a.dtype)
            cols.extend(
                (
                    p
                    * jnp.exp(-jnp.minimum(lam[pair] * layer_tau * fraction, 700))[
                        ..., None
                    ],
                    n
                    * jnp.exp(
                        -jnp.minimum(lam[pair] * layer_tau * (1 - fraction), 700)
                    )[..., None],
                )
            )
        eigen = jnp.stack(cols, axis=-1)
        cl = jnp.sqrt(a[2] * a[3]) / 3
        const = jnp.stack(
            (
                jnp.ones_like(cl),
                jnp.zeros_like(cl),
                jnp.zeros_like(cl),
                jnp.zeros_like(cl),
            ),
            -1,
        )
        poly = jnp.stack(
            (
                a[1] * layer_tau * fraction,
                jnp.ones_like(cl),
                jnp.zeros_like(cl),
                jnp.zeros_like(cl),
            ),
            -1,
        )
        decay = (
            jnp.stack(
                (
                    -2 * jnp.ones_like(cl),
                    jnp.zeros_like(cl),
                    jnp.ones_like(cl),
                    -3 * cl / a[3],
                ),
                -1,
            )
            * jnp.exp(-jnp.minimum(cl * layer_tau * fraction, 700))[..., None]
        )
        grow = (
            jnp.stack(
                (
                    -2 * jnp.ones_like(cl),
                    jnp.zeros_like(cl),
                    jnp.ones_like(cl),
                    3 * cl / a[3],
                ),
                -1,
            )
            * jnp.exp(-jnp.minimum(cl * layer_tau * (1 - fraction), 700))[..., None]
        )
        replacement = jnp.stack((const, poly, decay, grow), -1)
        return (
            jnp.where(conservative[..., None, None], replacement, eigen),
            lam,
            conservative,
        )

    def band_solve(band, rhs):
        """Bandwidth-five partial-pivot LU, independently vectorized in columns."""
        size, columns = rhs.shape
        lower, upper, diagonal = 5, 5, 10
        # This is LAPACK's expanded *band* storage: 16 x (4L) x columns,
        # never (4L) x (4L).  It is a direct JAX transcription of the CPU LU.
        expanded = jnp.zeros((16, size, columns), dtype=rhs.dtype)
        expanded = expanded.at[5:16].set(band)
        original = band
        vector = rhs
        pivots = jnp.zeros((size, columns), dtype=jnp.int32)
        column_index = jnp.arange(columns)

        def factor_column(column, carry):
            work, pivot_table = carry
            below = jnp.minimum(lower, size - column - 1)
            candidate_rows = diagonal + jnp.arange(lower + 1)
            candidates = jnp.abs(work[candidate_rows, column])
            candidates = jnp.where(
                jnp.arange(lower + 1)[:, None] <= below,
                candidates,
                -jnp.inf,
            )
            pivot_offset = jnp.argmax(candidates, axis=0).astype(jnp.int32)
            pivot_table = pivot_table.at[column].set(pivot_offset)
            last_offset = upper + below

            def swap_target(target_offset, value):
                def swap(active_value):
                    target = column + target_offset
                    first = diagonal - target_offset
                    second = first + pivot_offset
                    current = active_value[first, target]
                    replacement = active_value[second, target, column_index]
                    active_value = active_value.at[first, target].set(replacement)
                    return active_value.at[
                        second, jnp.full(columns, target), column_index
                    ].set(current)

                return lax.cond(
                    target_offset <= last_offset,
                    swap,
                    lambda inactive_value: inactive_value,
                    value,
                )

            work = lax.fori_loop(0, upper + lower + 1, swap_target, work)
            pivot = work[diagonal, column]

            def divide_offset(offset_index, value):
                offset = offset_index + 1
                return lax.cond(
                    offset <= below,
                    lambda active_value: active_value.at[diagonal + offset, column].set(
                        active_value[diagonal + offset, column] / pivot
                    ),
                    lambda inactive_value: inactive_value,
                    value,
                )

            work = lax.fori_loop(0, lower, divide_offset, work)

            def eliminate_target(target_index, value):
                target_offset = target_index + 1

                def eliminate(active_value):
                    target = column + target_offset
                    top = active_value[diagonal - target_offset, target]

                    def eliminate_offset(offset_index, inner_value):
                        offset = offset_index + 1

                        def update(update_value):
                            row = diagonal + offset - target_offset
                            return update_value.at[row, target].add(
                                -update_value[diagonal + offset, column] * top
                            )

                        return lax.cond(
                            offset <= below,
                            update,
                            lambda inactive_value: inactive_value,
                            inner_value,
                        )

                    return lax.fori_loop(0, lower, eliminate_offset, active_value)

                return lax.cond(
                    target_offset <= last_offset,
                    eliminate,
                    lambda inactive_value: inactive_value,
                    value,
                )

            work = lax.fori_loop(0, upper + lower, eliminate_target, work)
            return work, pivot_table

        expanded, pivots = lax.fori_loop(0, size, factor_column, (expanded, pivots))

        def forward_column(column, value):
            pivot = column + pivots[column]
            current = value[column]
            replacement = value[pivot, column_index]
            value = value.at[column].set(replacement)
            value = value.at[pivot, column_index].set(current)
            below = jnp.minimum(lower, size - column - 1)

            def forward_offset(offset_index, inner_value):
                offset = offset_index + 1
                return lax.cond(
                    offset <= below,
                    lambda active_value: active_value.at[column + offset].add(
                        -expanded[diagonal + offset, column] * active_value[column]
                    ),
                    lambda inactive_value: inactive_value,
                    inner_value,
                )

            return lax.fori_loop(0, lower, forward_offset, value)

        vector = lax.fori_loop(0, size - 1, forward_column, vector)

        def backward_iteration(iteration, value):
            column = size - 1 - iteration
            value = value.at[column].set(value[column] / expanded[diagonal, column])
            above = jnp.minimum(diagonal, column)

            def backward_offset(offset_index, inner_value):
                offset = offset_index + 1
                return lax.cond(
                    offset <= above,
                    lambda active_value: active_value.at[column - offset].add(
                        -expanded[diagonal - offset, column] * active_value[column]
                    ),
                    lambda inactive_value: inactive_value,
                    inner_value,
                )

            return lax.fori_loop(0, diagonal, backward_offset, value)

        vector = lax.fori_loop(0, size, backward_iteration, vector)
        solution = vector
        residual = -rhs
        row_norm = jnp.zeros((size, columns), dtype=rhs.dtype)

        def residual_column(column, carry):
            residual_value, norm_value = carry

            def residual_band(brow, inner_carry):
                residual_inner, norm_inner = inner_carry
                row = column + brow - upper

                def update(active_carry):
                    active_residual, active_norm = active_carry
                    coefficient = original[brow, column]
                    active_residual = active_residual.at[row].add(
                        coefficient * solution[column]
                    )
                    active_norm = active_norm.at[row].add(jnp.abs(coefficient))
                    return active_residual, active_norm

                return lax.cond(
                    (row >= 0) & (row < size),
                    update,
                    lambda inactive_carry: inactive_carry,
                    (residual_inner, norm_inner),
                )

            return lax.fori_loop(
                0, lower + upper + 1, residual_band, (residual_value, norm_value)
            )

        residual, row_norm = lax.fori_loop(
            0, size, residual_column, (residual, row_norm)
        )
        matrix_norm = jnp.max(row_norm, axis=0)
        denominator = matrix_norm * jnp.max(jnp.abs(solution), axis=0) + jnp.max(
            jnp.abs(rhs), axis=0
        )
        backward_error = jnp.where(
            denominator > 0,
            jnp.max(jnp.abs(residual), axis=0) / denominator,
            jnp.max(jnp.abs(residual), axis=0),
        )
        return solution, backward_error

    @jax.jit
    def run(tau, omega, moments, planck, bottom, mu, aw, gw, nodes, node_weights):
        nl, ns, ng = tau.shape
        nc = ns * ng
        tf = tau.reshape(nl, nc)
        of = omega.reshape(nl, nc)
        mf = moments.reshape(4, nl, nc)
        a = (2 * jnp.arange(4, dtype=tau.dtype) + 1)[:, None, None] - of[None] * mf
        top, _, _ = modes(a, tf, 0.0)
        bot, _, _ = modes(a, tf, 1.0)
        level_difference = jnp.broadcast_to(
            planck[1:, :, None] - planck[:-1, :, None], tau.shape
        ).reshape(nl, nc)
        slope = jnp.where(tf > 0, level_difference / tf, 0)
        thermal = jnp.where(a[0] > 0, (1 - of) / a[0], 0)
        pt = jnp.stack(
            (
                thermal
                * jnp.broadcast_to(planck[:-1, :, None], tau.shape).reshape(nl, nc),
                thermal * slope / a[1],
                jnp.zeros_like(tf),
                jnp.zeros_like(tf),
            ),
            -1,
        )
        pb = jnp.stack(
            (
                thermal
                * jnp.broadcast_to(planck[1:, :, None], tau.shape).reshape(nl, nc),
                thermal * slope / a[1],
                jnp.zeros_like(tf),
                jnp.zeros_like(tf),
            ),
            -1,
        )
        ft = half(top)
        fb = half(bot)
        fpt = half(pt[..., None])[..., 0]
        fpb = half(pb[..., None])[..., 0]
        size = 4 * nl
        band = jnp.zeros((11, size, nc), tau.dtype)
        rhs = jnp.zeros((size, nc), tau.dtype)

        def put(b, row, col, block):
            for i in range(block.shape[1]):
                for q in range(block.shape[2]):
                    b = b.at[5 + row + i - (col + q), col + q].set(block[:, i, q])
            return b

        band = put(band, 0, 0, ft[0, :, :2, :])
        rhs = rhs.at[:2].set(-fpt[0, :, :2].T)
        for layer in range(nl - 1):
            row = 2 + 4 * layer
            band = put(band, row, 4 * layer, fb[layer])
            band = put(band, row, 4 * (layer + 1), -ft[layer + 1])
            rhs = rhs.at[row : row + 4].set((fpt[layer + 1] - fpb[layer]).T)
        band = put(band, size - 2, size - 4, fb[-1, :, 2:, :])
        target = jnp.stack((jnp.pi * bottom, -0.25 * jnp.pi * bottom))
        rhs = rhs.at[-2:].set(
            jnp.repeat(target[:, :, None], ng, axis=2).reshape(2, nc) - fpb[-1, :, 2:].T
        )
        solved, backward_error = band_solve(band, rhs)
        # Keep the failure visible in the device result without introducing a
        # host synchronization; callers validate this through parity fixtures.
        solved = jnp.where(jnp.all(backward_error <= 2.0e-5), solved, jnp.nan)
        coeff = solved.reshape(nl, 4, nc).transpose(0, 2, 1)
        cumulative = jnp.cumsum(tau, axis=0) - tau
        total = jnp.sum(tau, axis=0)
        output = jnp.zeros(ns, tau.dtype)
        for node, weight in zip(nodes, node_weights):
            eig, _, _ = modes(a, tf, node)
            local = jnp.einsum("lcij,lcj->lci", eig, coeff)
            local = local.at[..., 0].add(
                thermal
                * (
                    jnp.broadcast_to(planck[:-1, :, None], tau.shape).reshape(nl, nc)
                    + node * slope * tf
                )
            )
            local = local.at[..., 1].add(thermal * slope / a[1])
            local = local.reshape(nl, ns, ng, 4)
            lp = (
                jnp.broadcast_to(planck[:-1, :, None], tau.shape)
                + node * slope.reshape(nl, ns, ng) * tau
            )
            for ai in range(mu.shape[0]):
                m = mu[ai]
                leg = jnp.array(
                    (1, m, 0.5 * (3 * m * m - 1), 0.5 * (5 * m * m * m - 3 * m)),
                    tau.dtype,
                )
                source = (
                    omega
                    * jnp.sum(local * moments.transpose(1, 2, 3, 0) * leg, axis=-1)
                    + (1 - omega) * lp
                )
                output = (
                    output
                    + aw[ai]
                    * gw
                    @ jnp.sum(
                        weight
                        * source
                        * jnp.exp(-(cumulative + node * tau) / m)
                        * tau
                        / m,
                        axis=0,
                    ).T
                )
        for ai in range(mu.shape[0]):
            output = output + aw[ai] * gw @ (
                (bottom[:, None] * jnp.exp(-total / mu[ai])).T
            )
        return jnp.maximum(output, 0)

    return run


__all__ = ["metal_solve_thermal_sh4_spectrum"]
