"""Strict runtime policy for the experimental Apple-Metal backend."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from robert_exoplanets.core import RobertConfigError


@dataclass(frozen=True)
class MetalRuntime:
    """One explicitly selected JAX accelerator device and array namespace.

    The historical name is retained for the public Apple-Metal API; CUDA
    callers can use the ``JaxAcceleratorRuntime`` alias below.
    """

    jax: Any
    jnp: Any
    device: Any
    accelerator: str | None = None

    @property
    def dtype(self):
        """Metal's supported scientific working type for this experiment."""

        return self.jnp.float32

    def put(self, value: Any):
        """Place a value on this Metal device without a host result transfer."""

        return self.jax.device_put(
            self.jnp.asarray(value, dtype=self.dtype), self.device
        )

    @property
    def metadata(self) -> dict[str, str]:
        """Return reproducibility metadata for manifests and benchmarks."""

        def package_version(name: str) -> str:
            try:
                return version(name)
            except PackageNotFoundError:
                return "unavailable"

        accelerator = (
            self.accelerator or str(getattr(self.device, "platform", "unknown")).lower()
        )
        return {
            "backend": f"jax-{accelerator}",
            "platform": str(getattr(self.device, "platform", "unknown")),
            "device_kind": str(getattr(self.device, "device_kind", "unknown")),
            "dtype": "float32",
            "jax_version": package_version("jax"),
            "jaxlib_version": package_version("jaxlib"),
            "jax_metal_version": package_version("jax-metal"),
        }

    @property
    def backend_query(self) -> str:
        """Return the JAX spelling for the selected accelerator backend."""

        accelerator = (
            self.accelerator or str(getattr(self.device, "platform", "unknown")).lower()
        )
        return {"metal": "METAL", "cuda": "gpu"}.get(accelerator, accelerator)

    @property
    def visible_device_count(self) -> int:
        """Count selected-backend devices without enabling device fan-out."""

        return len(tuple(self.jax.devices(self.backend_query)))


def require_metal_runtime() -> MetalRuntime:
    """Load JAX and return Metal only; never silently select CPU.

    Apple JAX Metal presently cannot execute ROBERT's float64/complex128
    reference path.  This isolated backend is therefore explicitly float32
    and experimental.  A float64-enabled JAX process is rejected rather than
    accidentally presenting a misleading precision mode.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - installation dependent.
        raise RobertConfigError(
            "the Metal backend requires JAX and Apple's jax-metal plug-in"
        ) from exc
    if jax.config.x64_enabled:
        raise RobertConfigError(
            "the Metal backend is float32-only; start Python without JAX_ENABLE_X64=1"
        )
    try:
        devices = tuple(jax.devices("METAL"))
    except (KeyError, RuntimeError, TypeError, ValueError):
        try:
            devices = tuple(jax.devices())
        except RuntimeError as exc:  # pragma: no cover - plugin initialization.
            raise RobertConfigError("JAX could not enumerate a Metal device") from exc
    metal_devices = tuple(
        device
        for device in devices
        if str(getattr(device, "platform", "")).lower() == "metal"
    )
    if not metal_devices:
        visible = ", ".join(
            f"{getattr(device, 'platform', '?')}:{getattr(device, 'device_kind', '?')}"
            for device in devices
        )
        raise RobertConfigError(
            "requested Metal backend is unavailable; refusing CPU fallback. "
            f"Visible JAX devices: {visible or 'none'}"
        )
    return MetalRuntime(jax=jax, jnp=jnp, device=metal_devices[0], accelerator="metal")


def require_accelerator_runtime(platform: str) -> MetalRuntime:
    """Select exactly one explicitly requested JAX accelerator device.

    The numerical graphs are platform-neutral JAX.  This helper makes the
    CUDA comparison opt-in while deliberately selecting device zero only: a
    benchmark must never silently fan out across every visible GPU.
    Apple callers should continue to use :func:`require_metal_runtime`, which
    retains its stricter Metal-specific diagnostics.
    """

    requested = str(platform).strip().lower()
    if not requested:
        raise RobertConfigError("an accelerator platform must be specified")
    if requested == "metal":
        return require_metal_runtime()
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RobertConfigError("the accelerator backend requires JAX") from exc
    backend_query = "gpu" if requested == "cuda" else requested
    try:
        devices = tuple(jax.devices(backend_query))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RobertConfigError(
            f"requested JAX accelerator platform is unavailable: {requested}"
        ) from exc
    matching = tuple(
        device
        for device in devices
        if str(getattr(device, "platform", "")).lower() == backend_query
    )
    if not matching:
        raise RobertConfigError(
            f"requested JAX accelerator platform is unavailable: {requested}"
        )
    if requested == "cuda":
        description = " ".join(
            (
                str(getattr(matching[0], "device_kind", "")),
                str(
                    getattr(
                        getattr(matching[0], "client", None),
                        "platform_version",
                        "",
                    )
                ),
            )
        ).lower()
        if "nvidia" not in description and "cuda" not in description:
            raise RobertConfigError(
                "requested CUDA but the visible JAX GPU is not identified as CUDA/NVIDIA"
            )
    return MetalRuntime(
        jax=jax,
        jnp=jnp,
        device=matching[0],
        accelerator=requested,
    )


# Platform-neutral spelling for new code.  Keep ``MetalRuntime`` stable for
# the existing Apple-facing experiment.
JaxAcceleratorRuntime = MetalRuntime


def clear_metal_caches() -> None:
    """Release ROBERT-held compiled kernels after a completed retrieval.

    Live arrays retained by user code remain live.  This function bounds
    long-running notebook/process growth when several unrelated static shapes
    have been compiled sequentially.
    """

    import gc

    from . import forward, mie, random_overlap, sh4

    mie._KERNELS.clear()
    sh4._KERNELS.clear()
    random_overlap._kernel.cache_clear()
    forward._compile_outer.cache_clear()
    try:
        import jax

        jax.clear_caches()
    except ImportError:  # pragma: no cover - optional dependency absent
        pass
    gc.collect()


__all__ = [
    "JaxAcceleratorRuntime",
    "MetalRuntime",
    "clear_metal_caches",
    "require_accelerator_runtime",
    "require_metal_runtime",
]
