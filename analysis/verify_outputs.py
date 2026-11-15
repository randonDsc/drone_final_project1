"""Validate a generated bundle and reproduce its numeric results from small files."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile

from .archive_io import json_write, sha256_file
from .run_analysis import analyze


def verify(output):
    output = Path(output).resolve()
    code_root = Path(__file__).parent
    manifest = json.loads((output/"outputs_manifest.json").read_text())
    actual_files = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if actual_files - {"outputs_manifest.json", "verification.json"} != set(manifest):
        raise ValueError("Output files differ from the generation manifest")
    for name, entry in manifest.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Invalid manifest path")
        saved = output/name
        if saved.stat().st_size != entry["bytes"] or sha256_file(saved) != entry["sha256"]:
            raise ValueError("Output checksum mismatch: "+name)

    provenance = json.loads((output/"provenance.json").read_text())
    current_sources = {p.name:sha256_file(p) for p in sorted(code_root.glob("*.py"))}
    if current_sources != provenance["analysis_sources_sha256"]:
        raise ValueError("Analysis source code changed since the saved run; regenerate into a new output folder")

    test = subprocess.run([sys.executable, "-m", "pytest", str(code_root/"tests"), "-q"],
                          cwd=code_root.parent, text=True, capture_output=True)
    if test.returncode:
        raise ValueError("Analysis tests failed:\n"+test.stdout+test.stderr)

    with tempfile.TemporaryDirectory(prefix="flight_statistics_verify_") as temp:
        reproduced = Path(temp)/"reproduced"
        analyze(reproduced, source_data=output/"source_data", config=provenance["configuration"], make_figures=False)
        table_names = sorted(p.name for p in (output/"tables").glob("*.csv"))
        if table_names != sorted(p.name for p in (reproduced/"tables").glob("*.csv")):
            raise ValueError("Reproduction table sets differ")
        checks = {}
        for name in table_names:
            original_hash = sha256_file(output/"tables"/name)
            if sha256_file(reproduced/"tables"/name) != original_hash:
                raise ValueError("Numeric reproduction differs: "+name)
            checks[name] = original_hash
        if json.loads((reproduced/"summary.json").read_text()) != json.loads((output/"summary.json").read_text()):
            raise ValueError("Reproduced headline results differ")

    result = dict(verified_utc=datetime.now(timezone.utc).isoformat(), status="passed",
                  output_manifest_sha256=sha256_file(output/"outputs_manifest.json"),
                  generation_files_verified=len(manifest), analysis_source_hashes_match=True,
                  test_command="python -m pytest analysis/tests -q", test_return_code=test.returncode,
                  test_output=test.stdout.strip(),
                  byte_identical_reproduced_tables=checks, summary_reproduced_exactly=True,
                  reproduction_input="preserved source_data only; no full archive or simulator",
                  full_archive_rehashed_by_verifier=False,
                  note="Verification record is intentionally outside the generation manifest to avoid circular hashes.")
    json_write(output/"verification.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.output)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Verified {result['generation_files_verified']} generated files and {len(result['byte_identical_reproduced_tables'])} byte-identical numeric tables.")
    print(result["test_output"])


if __name__ == "__main__":
    main()
