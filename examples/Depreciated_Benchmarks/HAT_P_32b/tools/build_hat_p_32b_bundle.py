"""Build the repository-local HAT-P-32b benchmark data bundle.

This maintainer utility needs the original NemesisPy reference products,
FastChem source tree, and native ExoMolOP KTA files. Normal ROBERT users do
not run it: the generated, exo_k-binned products are committed to the repo.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robert_exoplanets import (  # noqa: E402
    GridCoverage,
    OpacityDatabase,
    OpacityDataProduct,
    OpacityDataSource,
    OpacityMode,
    OpacityStorageFormat,
    SpectralCoverage,
    build_parameterized_clear_sky_emission_model,
    load_emission_observation_npz,
    write_robert_npz_archive,
)

from examples.hat_p_32b_fastchem_config import (  # noqa: E402
    OPACITY_SPECIES,
    make_model_config,
)

REFERENCE_FILES = (
    "quench_study_emission_G395H_spectra_band.npz",
    "quench_study_emission_corner_data.npz",
    "quench_study_emission_corner_meta.json",
    "quench_study_emission_TP_band.npz",
    "quench_study_emission_TP_VMR_band.npz",
    "quench_study_emission_summary.txt",
)
FASTCHEM_FILES = (
    "input/README.TXT",
    "input/element_abundances/README.TXT",
    "input/element_abundances/asplund_2009.dat",
    "input/logK/README.TXT",
    "input/logK/logK.dat",
    "licence.md",
)


def main() -> None:
    args = _parser().parse_args()
    output_root = Path(args.output_root)
    reference_dir = output_root / "reference"
    fastchem_output = output_root / "fastchem"
    opacity_dir = output_root / "opacities"
    for directory in (reference_dir, fastchem_output, opacity_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_reference = Path(args.reference_dir).expanduser()
    source_fastchem = Path(args.fastchem_dir).expanduser()
    source_kta = Path(args.kta_dir).expanduser()
    for filename in REFERENCE_FILES:
        shutil.copy2(source_reference / filename, reference_dir / filename)
    for filename in FASTCHEM_FILES:
        destination = fastchem_output / (
            "FASTCHEM_LICENSE.md" if filename == "licence.md" else filename
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_fastchem / filename, destination)

    observation = load_emission_observation_npz(
        reference_dir / "quench_study_emission_G395H_spectra_band.npz",
        instrument="JWST/NIRSpec G395H",
    )
    model = build_parameterized_clear_sky_emission_model(
        make_model_config(
            kta_dir=source_kta,
            fastchem_path=source_fastchem,
        ),
        spectral_grid=observation.spectral_grid,
    )
    for species in OPACITY_SPECIES:
        table = model.opacity_provider.tables[species]
        source_metadata = dict(table.metadata)
        source_checksum = source_metadata.get("checksum_sha256", "")
        product = OpacityDataProduct(
            species=(species,),
            mode=OpacityMode.CORRELATED_K,
            source=OpacityDataSource.EXOMOL_OP,
            storage_format=OpacityStorageFormat.KTA_BINARY,
            spectral_coverage=SpectralCoverage(
                min_value=float(table.wavenumber_cm_inverse.min()),
                max_value=float(table.wavenumber_cm_inverse.max()),
                unit="cm^-1",
                n_points=table.wavenumber_cm_inverse.size,
            ),
            grid_coverage=GridCoverage(
                pressure_min=float(table.pressure_bar.min()),
                pressure_max=float(table.pressure_bar.max()),
                pressure_unit="bar",
                temperature_min=float(table.temperature_K.min()),
                temperature_max=float(table.temperature_K.max()),
                temperature_unit="K",
                n_pressure=table.pressure_bar.size,
                n_temperature=table.temperature_K.size,
            ),
            g_ordinates=table.g_samples.size,
            checksum_sha256=source_checksum or None,
            native_shape=table.kcoeff.shape,
            metadata={
                key: str(value)
                for key, value in source_metadata.items()
                if key not in {"source_path", "checksum_sha256"}
            }
            | {
                "kcoeff_unit": table.unit,
                "source_kta_filename": f"{species}_emission_R1000.kta",
                "source_kta_sha256": source_checksum,
                "benchmark_spectral_grid": "JWST/NIRSpec G395H HAT-P-32b",
            },
        )
        database = OpacityDatabase(
            products=(product,),
            name=f"HAT-P-32b-{species}-exo-k-binned",
            metadata={
                "benchmark": "HAT-P-32b FastChem/Madhusudhan-Seager retrieval",
                "spectral_preparation": "exo_k_bin_down",
            },
        )
        output_path = opacity_dir / f"{species}.robert-opacity.npz"
        write_robert_npz_archive(
            output_path,
            database=database,
            arrays={
                "kcoeff": table.kcoeff,
                "pressure_bar": table.pressure_bar,
                "temperature_K": table.temperature_K,
                "wavenumber_cm-1": table.wavenumber_cm_inverse,
                "wavelength_micron": table.wavelength_micron,
                "g_samples": table.g_samples,
                "g_weights": table.g_weights,
            },
            compressed=True,
            metadata={
                "benchmark": "HAT-P-32b",
                "generated_by": "tools/build_hat_p_32b_bundle.py",
            },
        )

    provenance = {
        "bundle": "HAT-P-32b FastChem/Madhusudhan-Seager retrieval benchmark",
        "opacity_preparation": "exo_k bin_down_cp with 300 samples per target bin",
        "fastchem_runtime": "pyfastchem 3.1.2",
        "fastchem_source": "https://github.com/NewStrangeWorlds/FastChem",
        "fastchem_license": "GPL-3.0",
        "cia_source": "NemesisPy v1.0.1 (packaged under robert_exoplanets/data/cia)",
        "reference_products": list(REFERENCE_FILES),
        "opacity_species": list(OPACITY_SPECIES),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(output_root)}")
    (output_root / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--kta-dir", required=True)
    parser.add_argument("--fastchem-dir", required=True)
    parser.add_argument(
        "--output-root",
        default="examples/data/hat_p_32b",
    )
    return parser


if __name__ == "__main__":
    main()
