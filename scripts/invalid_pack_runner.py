#!/usr/bin/env python3
"""Run invalid XML files; assert parser reports Invalid XML on stderr."""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from xml_debug_pack_common import first_root_xpath_from_xml_snippet, strip_ansi


def run_case(exe, xml_path, xpath, timeout):
    # type: (Path, Path, str, float) -> Tuple[str, str, int]
    proc = subprocess.run(
        [str(exe), str(xml_path), xpath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=timeout,
    )
    return proc.stdout, proc.stderr, proc.returncode


def main():
    # type: () -> int
    ap = argparse.ArgumentParser(description="Run invalid XML debug pack (10 files).")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="XML-Parser repository root",
    )
    ap.add_argument("--exe", type=Path, default=Path("output.exe"), help="Path to output.exe")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="queries_manifest.json",
    )
    ap.add_argument(
        "--invalid-dir",
        type=Path,
        default=None,
        help="Directory with invalid XML files",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Report directory",
    )
    ap.add_argument("--timeout", type=float, default=300.0, help="Per-run timeout (seconds)")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    manifest = (args.manifest or repo / "dataset" / "xml_debug_pack" / "queries_manifest.json").resolve()
    invalid_dir = (args.invalid_dir or repo / "dataset" / "xml_debug_pack" / "invalid_xml").resolve()
    out_dir = (args.out_dir or repo / "dataset" / "xml_debug_pack" / "reports").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = args.exe if args.exe.is_absolute() else (repo / args.exe).resolve()
    if not exe.is_file():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 2

    with open(manifest, encoding="utf-8") as f:
        data = json.load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"invalid_run_{ts}.log"
    csv_path = out_dir / f"invalid_results_{ts}.csv"

    rows = []  # type: List[Dict[str, str]]
    failed = 0

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# invalid XML debug pack run {ts}\n")
        logf.write(f"# exe={exe}\n\n")

        for inv in data["invalid_files"]:
            filename = inv["filename"]
            xml_path = invalid_dir / filename
            xpath = first_root_xpath_from_xml_snippet(inv.get("xml", ""))
            banner = f"=== {filename} :: xpath {xpath} ===\n"
            logf.write(banner)

            if not xml_path.is_file():
                logf.write(f"MISSING FILE: {xml_path}\n\n")
                rows.append(
                    {
                        "xml_file": filename,
                        "xpath": xpath,
                        "error_detected": "no",
                        "stderr_excerpt": f"missing file: {xml_path}",
                    }
                )
                failed += 1
                continue

            try:
                stdout, stderr, code = run_case(exe, xml_path, xpath, args.timeout)
            except subprocess.TimeoutExpired:
                logf.write("TIMEOUT\n\n")
                rows.append(
                    {
                        "xml_file": filename,
                        "xpath": xpath,
                        "error_detected": "no",
                        "stderr_excerpt": "timeout",
                    }
                )
                failed += 1
                continue

            logf.write("--- stdout ---\n")
            logf.write(stdout)
            if not stdout.endswith("\n"):
                logf.write("\n")
            logf.write("--- stderr ---\n")
            logf.write(stderr)
            if not stderr.endswith("\n"):
                logf.write("\n")
            logf.write(f"--- exit {code} ---\n\n")

            err_plain = strip_ansi(stderr)
            detected = "Invalid XML" in err_plain
            excerpt = err_plain.strip().replace("\n", " ")[:300]

            rows.append(
                {
                    "xml_file": filename,
                    "xpath": xpath,
                    "error_detected": "yes" if detected else "no",
                    "stderr_excerpt": excerpt,
                }
            )
            if not detected:
                failed += 1

    fieldnames = ["xml_file", "xpath", "error_detected", "stderr_excerpt"]
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {log_path}")
    print(f"Wrote {csv_path} ({len(rows)} rows)")
    if failed:
        print(f"Summary: {failed} file(s) did not report Invalid XML on stderr", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
