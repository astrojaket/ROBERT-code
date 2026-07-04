"""Benchmark candidate ROBERT opacity archive read formats."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from robert_exoplanets import (
    GridCoverage,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityDatabase,
    OpacityMode,
    OpacityStorageFormat,
    SpectralCoverage,
    load_robert_npy_directory,
    load_robert_npz_archive,
    time_callable,
    write_robert_npy_directory,
    write_robert_npz_archive,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "opacity_archive_benchmark"


def main() -> None:
    repeat = int(os.environ.get("ROBERT_OPACITY_IO_REPEAT", "50"))
    warmup = int(os.environ.get("ROBERT_OPACITY_IO_WARMUP", "5"))
    n_pressure = int(os.environ.get("ROBERT_OPACITY_IO_NP", "6"))
    n_temperature = int(os.environ.get("ROBERT_OPACITY_IO_NT", "7"))
    n_wavelength = int(os.environ.get("ROBERT_OPACITY_IO_NWAVE", "128"))
    n_g = int(os.environ.get("ROBERT_OPACITY_IO_NG", "8"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database = _synthetic_database(n_pressure, n_temperature, n_wavelength, n_g)
    arrays = {
        "kcoeff": _synthetic_kcoeff(n_pressure, n_temperature, n_wavelength, n_g),
        "g_weights": np.full(n_g, 1.0 / n_g, dtype=np.float64),
        "wavenumber_cm-1": np.linspace(1000.0, 5000.0, n_wavelength, dtype=np.float64),
        "pressure_bar": np.logspace(-6.0, 2.0, n_pressure, dtype=np.float64),
        "temperature_K": np.linspace(300.0, 2500.0, n_temperature, dtype=np.float64),
    }

    native_dir = OUTPUT_DIR / "synthetic.robert-opacity"
    uncompressed_npz = OUTPUT_DIR / "synthetic_uncompressed.npz"
    compressed_npz = OUTPUT_DIR / "synthetic_compressed.npz"
    write_robert_npy_directory(native_dir, database=database, arrays=arrays, overwrite=True)
    write_robert_npz_archive(uncompressed_npz, database=database, arrays=arrays, compressed=False)
    write_robert_npz_archive(compressed_npz, database=database, arrays=arrays, compressed=True)

    read_targets = {
        "npy-directory-eager": lambda: load_robert_npy_directory(native_dir),
        "npy-directory-mmap": lambda: load_robert_npy_directory(native_dir, mmap_mode="r"),
        "npz-uncompressed": lambda: load_robert_npz_archive(uncompressed_npz),
        "npz-compressed": lambda: load_robert_npz_archive(compressed_npz),
    }

    print("Benchmark: opacity archive I/O")
    print(f"Shape: pressure={n_pressure}, temperature={n_temperature}, wavelength={n_wavelength}, g={n_g}")
    print(f"Repeats: {repeat}")
    print(f"Warmups: {warmup}")
    for name, function in read_targets.items():
        result = time_callable(function, name=name, repeat=repeat, warmup=warmup)
        print(
            f"- {name}: median={result.median_s * 1.0e3:.3f} ms, "
            f"best={result.min_s * 1.0e3:.3f} ms, calls/s={result.calls_per_second:.1f}"
        )


def _synthetic_database(
    n_pressure: int,
    n_temperature: int,
    n_wavelength: int,
    n_g: int,
) -> OpacityDatabase:
    product = OpacityDataProduct(
        species=("H2O",),
        mode=OpacityMode.CORRELATED_K,
        source=OpacityDataSource.EXOMOL_OP,
        storage_format=OpacityStorageFormat.NEMESIS_KTA,
        spectral_coverage=SpectralCoverage(1000.0, 5000.0, unit="cm^-1", n_points=n_wavelength),
        grid_coverage=GridCoverage(
            pressure_min=1.0e-6,
            pressure_max=100.0,
            temperature_min=300.0,
            temperature_max=2500.0,
            n_pressure=n_pressure,
            n_temperature=n_temperature,
        ),
        g_ordinates=n_g,
        native_shape=(n_pressure, n_temperature, n_wavelength, n_g),
    )
    return OpacityDatabase(products=(product,), name="synthetic-opacity-benchmark")


def _synthetic_kcoeff(
    n_pressure: int,
    n_temperature: int,
    n_wavelength: int,
    n_g: int,
) -> np.ndarray:
    pressure_axis = np.linspace(0.0, 1.0, n_pressure)[:, None, None, None]
    temperature_axis = np.linspace(0.0, 1.0, n_temperature)[None, :, None, None]
    wavelength_axis = np.linspace(0.0, 1.0, n_wavelength)[None, None, :, None]
    g_axis = np.linspace(0.0, 1.0, n_g)[None, None, None, :]
    return (
        1.0e-30
        + 1.0e-25 * pressure_axis
        + 1.0e-26 * temperature_axis
        + 1.0e-27 * wavelength_axis
        + 1.0e-28 * g_axis
    ).astype(np.float64)


if __name__ == "__main__":
    main()
