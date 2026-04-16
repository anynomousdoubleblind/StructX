#!/usr/bin/env python3
"""
Run StructX output.exe over a fixed (file, XPath) matrix and write a wide CSV
of timing breakdown (DEBUG_MODE=3 ms lines), token count, match count, and
optionally max GPU memory Used (MB) from DEBUG_MODE=5 via --two-pass.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
STRUCTX_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from xml_debug_pack_common import parse_matches_found, strip_ansi  # noqa: E402

# (column_header, xml path relative to StructX root, XPath query)
BENCHMARK_RUNS = [
    ("PSD[0]", "dataset/psd7003.xml", "/ProteinDatabase/ProteinEntry/reference/refinfo/authors/author"),
    ("PSD[1]", "dataset/psd7003.xml", "/ProteinDatabase/ProteinEntry/reference/refinfo[@refid='A31764']/authors/author"),
    ("PSD[2]", "dataset/psd7003.xml", "/ProteinDatabase/ProteinEntry/reference/refinfo[volume=85]/authors/author[0]"),
    ("NY_MOTOR[0]", "dataset/data_gov/ny_motor_vehicle_collisions_cutted_NEW.xml", "/response/row/row/crash_time/"),
    ("NY_MOTOR[1]", "dataset/data_gov/ny_motor_vehicle_collisions_cutted_NEW.xml", "/response/row/row[@_id='row-t9dc~q8a7~jwk4']/collision_id/"),
    ("CRIME_LA[0]", "dataset/data_gov/crime_la_2020_to_present_new.xml", "/response/row/row/area/"),
    ("CRIME_LA[1]", "dataset/data_gov/crime_la_2020_to_present_new.xml", "/response/row/row[status=IC]/area/"),
    ("CT_RE[0]", "dataset/data_gov/connecticut_real_state_2001_to_2022_new.xml", "/response/row/row/serialnumber/"),
    ("CAMPAIGN[0]", "dataset/data_gov/campaign_finance_report.xml", "/response/row/row/report_number/"),
    ("CAMPAIGN[1]", "dataset/data_gov/campaign_finance_report.xml", "/response/row/row[origin=C3]/report_number/"),
    ("TWITTER[0]", "dataset/json2xml/twitter.xml", "/root/item/entities/urls/"),
    ("TWITTER[1]", "dataset/json2xml/twitter.xml", "/root/item/entities/urls/item[0]/display_url/"),
    ("WALMART[0]", "dataset/json2xml/walmart.xml", "/root/items/item/salePrice/"),
    ("GOOGLE_MAP[0]", "dataset/json2xml/google_map.xml", "/root/item/routes/item/bounds/northeast/lng/"),
]

ROW_LABELS = [
    "host_to_device_ms",
    "validation_ms",
    "tokenization_ms",
    "parser_ms",
    "total_parsing_ms",
    "query_compute_ms",
    "query_d2h_ms",
    "total_query_ms",
    "total_end_to_end_ms",
    "tokens_count",
    "matches_found",
    "gpu_memory_used_mb_max",
]


def _parse_ms(stdout: str, pattern: str) -> Optional[float]:
    clean = strip_ansi(stdout)
    m = re.search(pattern, clean)
    if not m:
        return None
    return float(m.group(1))


def parse_host_to_device_ms(stdout: str) -> Optional[float]:
    return _parse_ms(stdout, r"Host to Device execution time:\s*([\d.]+)\s*ms")


def parse_validation_ms(stdout: str) -> Optional[float]:
    return _parse_ms(stdout, r"Validation execution time:\s*([\d.]+)\s*ms")


def parse_tokenization_ms(stdout: str) -> Optional[float]:
    return _parse_ms(stdout, r"Tokenization execution time:\s*([\d.]+)\s*ms")


def parse_parser_ms(stdout: str) -> Optional[float]:
    return _parse_ms(stdout, r"Parser execution time:\s*([\d.]+)\s*ms")


def parse_query_compute_ms(stdout: str) -> Optional[float]:
    return _parse_ms(stdout, r"Query Computation time:\s*([\d.]+)\s*ms")


def parse_query_d2h_ms(stdout: str) -> Optional[float]:
    return _parse_ms(stdout, r"Query Device to Host execution time:\s*([\d.]+)\s*ms")


def parse_tokens_count(stdout: str) -> Optional[int]:
    clean = strip_ansi(stdout)
    m = re.search(r"Tokens Count:\s*(\d+)", clean)
    return int(m.group(1)) if m else None


def parse_max_gpu_memory_used_mb(stdout: str) -> Optional[float]:
    """Max of all 'Used = X.XX MB' values from GPU Memory Usage lines (DEBUG_MODE=5)."""
    clean = strip_ansi(stdout)
    vals = [float(x) for x in re.findall(r"Used\s*=\s*([\d.]+)\s*MB", clean)]
    return max(vals) if vals else None


def parse_run(stdout):
    # type: (str) -> Dict[str, Any]
    h2d = parse_host_to_device_ms(stdout)
    val = parse_validation_ms(stdout)
    tok = parse_tokenization_ms(stdout)
    par = parse_parser_ms(stdout)
    qc = parse_query_compute_ms(stdout)
    d2h = parse_query_d2h_ms(stdout)
    tokens = parse_tokens_count(stdout)
    matches = parse_matches_found(stdout)
    gpu_mem_max = parse_max_gpu_memory_used_mb(stdout)

    total_parse = None
    if h2d is not None and tok is not None and par is not None:
        total_parse = h2d + tok + par
        if val is not None:
            total_parse += val

    total_query = None
    if qc is not None and d2h is not None:
        total_query = qc + d2h

    total_e2e = None
    if total_parse is not None and total_query is not None:
        total_e2e = total_parse + total_query

    return {
        "host_to_device_ms": h2d,
        "validation_ms": val,
        "tokenization_ms": tok,
        "parser_ms": par,
        "total_parsing_ms": total_parse,
        "query_compute_ms": qc,
        "query_d2h_ms": d2h,
        "total_query_ms": total_query,
        "total_end_to_end_ms": total_e2e,
        "tokens_count": tokens,
        "matches_found": matches,
        "gpu_memory_used_mb_max": gpu_mem_max,
    }


def format_cell(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def column_headers():
    # type: () -> list
    return [label for label, _, _ in BENCHMARK_RUNS]


def execute_all_runs(structx_root, exe, dry_run):
    # type: (Path, Path, bool) -> Dict[str, Dict[str, Any]]
    """Run every benchmark; parse full metrics from stdout."""
    results = {}  # type: Dict[str, Dict[str, Any]]
    for col, rel_xml, xpath in BENCHMARK_RUNS:
        xml_path = structx_root / rel_xml
        if not xml_path.is_file():
            print(f"warning: missing XML (skipping column {col}): {xml_path}", file=sys.stderr)
            results[col] = {k: None for k in ROW_LABELS}
            continue

        if dry_run:
            print(f"[dry-run] {col}: {xml_path} :: {xpath!r}")
            results[col] = {k: None for k in ROW_LABELS}
            continue

        print(f"running {col} ...", flush=True)
        try:
            proc = subprocess.run(
                [str(exe), str(xml_path), xpath],
                cwd=str(structx_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=False,
            )
        except OSError as e:
            print(f"error running {exe}: {e}", file=sys.stderr)
            results[col] = {k: None for k in ROW_LABELS}
            continue

        out = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0:
            print(
                f"warning: {col} exited {proc.returncode}; output may be incomplete",
                file=sys.stderr,
            )

        results[col] = parse_run(out)
    return results


def write_breakdown_csv(out_csv, results, columns):
    # type: (Path, Dict[str, Dict[str, Any]], list) -> None
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + columns)
        for row_key in ROW_LABELS:
            w.writerow([row_key] + [format_cell(results[c].get(row_key)) for c in columns])


def compile_with_debug_mode(structx_root, compile_script, mode):
    # type: (Path, Path, int) -> None
    env = os.environ.copy()
    env["DEBUG_MODE"] = str(mode)
    subprocess.run(
        ["bash", str(compile_script)],
        cwd=str(structx_root),
        env=env,
        check=True,
    )


def run_two_pass_collect(structx_root, exe, out_csv, compile_script, dry_run):
    # type: (Path, Path, Path, Path, bool) -> int
    """Compile DEBUG_MODE=3, run benchmarks; compile DEBUG_MODE=5, re-run for max GPU memory."""
    columns = column_headers()
    if not dry_run:
        print("phase 1: compile DEBUG_MODE=3 (timings)", flush=True)
        compile_with_debug_mode(structx_root, compile_script, 3)
        exe = (structx_root / "output.exe").resolve()
        if not exe.is_file():
            print(f"error: binary not found after compile: {exe}", file=sys.stderr)
            return 1

    results = execute_all_runs(structx_root, exe, dry_run)

    if dry_run:
        write_breakdown_csv(out_csv, results, columns)
        print(f"wrote {out_csv}")
        return 0

    print("phase 2: compile DEBUG_MODE=5 (GPU memory usage)", flush=True)
    compile_with_debug_mode(structx_root, compile_script, 5)
    exe = (structx_root / "output.exe").resolve()
    if not exe.is_file():
        print(f"error: binary not found after compile: {exe}", file=sys.stderr)
        return 1

    print("phase 2: re-running benchmarks (collect max GPU memory Used MB)", flush=True)
    for col, rel_xml, xpath in BENCHMARK_RUNS:
        xml_path = structx_root / rel_xml
        if not xml_path.is_file():
            if col in results:
                results[col]["gpu_memory_used_mb_max"] = None
            continue
        print(f"  memory sample {col} ...", flush=True)
        try:
            proc = subprocess.run(
                [str(exe), str(xml_path), xpath],
                cwd=str(structx_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=False,
            )
        except OSError as e:
            print(f"error running {exe}: {e}", file=sys.stderr)
            if col in results:
                results[col]["gpu_memory_used_mb_max"] = None
            continue
        out = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0:
            print(
                f"warning: {col} exited {proc.returncode}; memory line may be incomplete",
                file=sys.stderr,
            )
        gmax = parse_max_gpu_memory_used_mb(out)
        if col in results:
            results[col]["gpu_memory_used_mb_max"] = gmax

    write_breakdown_csv(out_csv, results, columns)
    print(f"wrote {out_csv}")
    return 0


def run_collect(
    structx_root: Path,
    exe: Path,
    out_csv: Path,
    dry_run: bool = False,
) -> int:
    if not exe.is_file():
        print(f"error: binary not found: {exe}", file=sys.stderr)
        return 1

    columns = column_headers()
    results = execute_all_runs(structx_root, exe, dry_run)
    write_breakdown_csv(out_csv, results, columns)
    print(f"wrote {out_csv}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Collect StructX algorithm breakdown into a CSV.")
    p.add_argument(
        "--structx-root",
        type=Path,
        default=STRUCTX_ROOT,
        help="StructX project root (default: parent of this script)",
    )
    p.add_argument(
        "--exe",
        type=Path,
        default=None,
        help="Path to output.exe (default: <structx-root>/output.exe)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=STRUCTX_ROOT / "results" / "algorithm_breakdown.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs only; write NA CSV without executing the binary",
    )
    p.add_argument(
        "--single-pass",
        action="store_true",
        help="One compile (DEBUG_MODE=3) and one benchmark set only (faster). "
        "gpu_memory_used_mb_max will be NA because GPU memory lines require DEBUG_MODE=5.",
    )
    p.add_argument(
        "--compile-script",
        type=Path,
        default=SCRIPT_DIR / "compile_algorithm_breakdown.sh",
        help="Compile script for the default two-pass run (default: scripts/compile_algorithm_breakdown.sh)",
    )
    args = p.parse_args()
    root = args.structx_root.resolve()
    exe = (args.exe or (root / "output.exe")).resolve()
    out = args.output.resolve()
    if args.single_pass:
        return run_collect(root, exe, out, dry_run=args.dry_run)
    return run_two_pass_collect(
        root,
        exe,
        out,
        args.compile_script.resolve(),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
