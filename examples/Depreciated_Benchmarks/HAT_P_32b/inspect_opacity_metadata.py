"""Inspect local opacity metadata without loading opacity arrays."""

from __future__ import annotations

import os
from pathlib import Path

from robert_exoplanets.opacity import OpacityDatabase, inspect_kta_file


DEFAULT_KTA_DIR = Path(
    "/Users/jaketaylor/Dropbox/PostDoc4/Emission_Example/HAT-P-32b/kta_temp"
)


def main() -> None:
    kta_dir = Path(os.environ.get("HAT_P_32B_KTA_DIR", str(DEFAULT_KTA_DIR))).expanduser()
    if not kta_dir.exists():
        print(f"Opacity directory not found: {kta_dir}")
        print("Set HAT_P_32B_KTA_DIR to inspect a different local k-table folder.")
        return

    products = tuple(inspect_kta_file(path) for path in sorted(kta_dir.glob("*.kta")))
    if not products:
        print(f"No .kta files found in {kta_dir}")
        return

    database = OpacityDatabase(
        products=products,
        name="HAT-P-32b ExoMolOP/exo_k k-tables",
        root=str(kta_dir),
    )
    print(f"Database: {database.name}")
    print(f"Root: {database.root}")
    print(f"Species: {', '.join(database.species)}")
    for product in database.products:
        size_mb = (product.file_size_bytes or 0) / 1024**2
        spectral = product.spectral_coverage
        grid = product.grid_coverage
        spectral_text = "coverage=unknown"
        if spectral is not None:
            spectral_text = f"{spectral.min_value:.1f}-{spectral.max_value:.1f} {spectral.unit}"
        grid_text = "P/T=unknown"
        if grid is not None:
            grid_text = (
                f"P={grid.pressure_min:.2e}-{grid.pressure_max:.2e} {grid.pressure_unit}, "
                f"T={grid.temperature_min:.0f}-{grid.temperature_max:.0f} {grid.temperature_unit}"
            )
        print(
            f"- {product.species[0]}: source={product.source.value}, "
            f"format={product.storage_format.value}, mode={product.mode.value}, "
            f"shape={product.native_shape}, {spectral_text}, {grid_text}, "
            f"size={size_mb:.1f} MiB, sha256={product.checksum_sha256[:12]}"
        )


if __name__ == "__main__":
    main()
