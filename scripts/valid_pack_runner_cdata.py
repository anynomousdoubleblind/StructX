#!/usr/bin/env python3
"""Run all valid_xml + XPath cases from queries_manifest.json; write one log and CSV."""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from xml_debug_pack_common import parse_match_zero_text, parse_matches_found, strip_ansi


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


def classify_error(stderr: str, stdout: str) -> str:
    s = strip_ansi(stderr).strip()
    o = strip_ansi(stdout).strip()
    if "Failed to load XML file" in s or "Failed to load XML file" in o:
        return "load_failed"
    if "Invalid XML" in s:
        return "validation_error"
    if "Error:" in s or "error" in s.lower():
        return s[:200].replace("\n", " ") if s else "stderr_present"
    return ""


def row_status(qtype, expected_count, expected_text, run_matches, stdout):
    # type: (str, int, Optional[str], Optional[int], str) -> str
    if run_matches is None:
        return "wrong"
    if qtype == "count":
        return "correct" if run_matches == expected_count else "wrong"
    # text
    if run_matches != 1:
        return "wrong"
    got = parse_match_zero_text(stdout)
    if got is None:
        return "wrong"
    if got == (expected_text or ""):
        return "correct"
    return "wrong"


def main():
    # type: () -> int
    ap = argparse.ArgumentParser(description="Run valid XML debug pack (100 cases).")
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
        help="queries_manifest.json (default: dataset/xml_debug_pack/queries_manifest.json)",
    )
    ap.add_argument(
        "--valid-dir",
        type=Path,
        default=None,
        help="Directory with valid XML files (default: dataset/xml_debug_pack/valid_xml_withCDATA)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Report directory (default: dataset/xml_debug_pack/reports under repo)",
    )
    ap.add_argument("--timeout", type=float, default=300.0, help="Per-run timeout (seconds)")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    manifest = (args.manifest or repo / "dataset" / "xml_debug_pack" / "queries_manifest.json").resolve()
    valid_dir = (args.valid_dir or repo / "dataset" / "xml_debug_pack" / "valid_xml_withCDATA").resolve()
    out_dir = (args.out_dir or repo / "dataset" / "xml_debug_pack" / "reports").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = args.exe if args.exe.is_absolute() else (repo / args.exe).resolve()
    if not exe.is_file():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 2

    with open(manifest, encoding="utf-8") as f:
        data = json.load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"valid_run_{ts}.log"
    csv_path = out_dir / f"valid_results_{ts}.csv"

    rows = []  # type: List[Dict[str, str]]

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# valid XML debug pack run {ts}\n")
        logf.write(f"# exe={exe}\n")
        logf.write(f"# manifest={manifest}\n\n")

        for vf in data["valid_files"]:
            filename = vf["filename"]
            xml_path = valid_dir / filename
            for qi, q in enumerate(vf["queries"], start=1):
                xpath = q["query"]
                qtype = q["type"]
                banner = (
                    f"=== {filename} :: query_index {qi} :: {xpath} ===\n"
                )
                logf.write(banner)

                if not xml_path.is_file():
                    logf.write(f"MISSING FILE: {xml_path}\n\n")
                    err = "missing_xml_file"
                    run_m = None
                    stdout = ""
                    stderr = f"File not found: {xml_path}"
                    status = "wrong"
                    readme_m = str(q["expected"]) if qtype == "count" else "1"
                    rows.append(
                        {
                            "xml_file": filename,
                            "query_path": xpath,
                            "readme_expected_matches": readme_m,
                            "run_matches": "ERROR",
                            "status": status,
                            "error": err,
                        }
                    )
                    continue

                try:
                    stdout, stderr, code = run_case(exe, xml_path, xpath, args.timeout)
                except subprocess.TimeoutExpired:
                    stdout = ""
                    stderr = "timeout"
                    run_m = None
                    err = "timeout"
                    logf.write("TIMEOUT\n\n")
                    readme_m = str(q["expected"]) if qtype == "count" else "1"
                    rows.append(
                        {
                            "xml_file": filename,
                            "query_path": xpath,
                            "readme_expected_matches": readme_m,
                            "run_matches": "ERROR",
                            "status": "wrong",
                            "error": err,
                        }
                    )
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

                run_m = parse_matches_found(stdout)
                err = classify_error(stderr, stdout)

                if qtype == "count":
                    readme_expected = int(q["expected"])
                    readme_m = str(readme_expected)
                else:
                    readme_expected = 1
                    readme_m = "1"

                status = row_status(qtype, readme_expected, q.get("expected"), run_m, stdout)
                run_m_str = str(run_m) if run_m is not None else "ERROR"

                rows.append(
                    {
                        "xml_file": filename,
                        "query_path": xpath,
                        "readme_expected_matches": readme_m,
                        "run_matches": run_m_str,
                        "status": status,
                        "error": err,
                    }
                )

    fieldnames = [
        "xml_file",
        "query_path",
        "readme_expected_matches",
        "run_matches",
        "status",
        "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {log_path}")
    print(f"Wrote {csv_path} ({len(rows)} rows)")
    wrong = sum(1 for r in rows if r["status"] == "wrong")
    if wrong:
        print(f"Summary: {wrong} wrong, {len(rows) - wrong} correct", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
