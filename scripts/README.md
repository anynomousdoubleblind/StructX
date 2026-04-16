# Algorithm breakdown scripts

These scripts build StructX, run a fixed set of XML plus XPath benchmarks, and write a wide CSV of timings and other metrics to `results/algorithm_breakdown.csv` (or a path you pass with `-o`).

## Scripts

| File | Role |
|------|------|
| `compile_algorithm_breakdown.sh` | Invokes `nvcc` from the StructX repo root. Sets `-DDEBUG_MODE` from the environment (default `3`). Use `SM` or `NVCC_GENCODE` to target a GPU architecture (default `sm_61`). |
| `collect_algorithm_breakdown.py` | Runs `output.exe` for each benchmark (see below), parses stdout/stderr, writes the CSV. **By default** it runs a **two-pass** pipeline (see next section). |
| `run_algorithm_breakdown.sh` | Convenience wrapper: calls `collect_algorithm_breakdown.py` with `--structx-root` set to the parent StructX directory. |
| `run_algorithm_breakdown_two_pass.sh` | Same behavior as `run_algorithm_breakdown.sh` (two-pass is the default). Kept as a stable alias. |

Helper module `xml_debug_pack_common.py` is shared with other runners (ANSI stripping, match parsing).

## How the collector runs (default: two-pass)

1. **Phase 1 — timings:** `compile_algorithm_breakdown.sh` is run with `DEBUG_MODE=3`. That build prints coarse stage timings in milliseconds (`⏱️ … ms` lines). The collector runs **all** benchmarks once and records those metrics plus token count and match count.

2. **Phase 2 — GPU memory:** The same script is run again with `DEBUG_MODE=5`. That build prints many `GPU Memory Usage: Used = … MB` lines during execution. The collector runs **the same** benchmarks again and, for each column, records the **maximum** `Used` value seen anywhere in that run’s output.

So you get **two `nvcc` builds** and **28 process runs** (14 XML/XPath pairs twice). Use `--single-pass` to skip phase 2 (one build, 14 runs): faster, but the `gpu_memory_used_mb_max` row will be `NA` because only `DEBUG_MODE=5` prints GPU memory snapshots.

### Common commands

From the StructX repository root:

```bash
# Full table (timings + peak GPU memory per column)
./scripts/run_algorithm_breakdown.sh -o results/algorithm_breakdown.csv

# Same, calling Python directly
python3 scripts/collect_algorithm_breakdown.py -o results/algorithm_breakdown.csv

# Timing-only; gpu_memory row stays NA
python3 scripts/collect_algorithm_breakdown.py --single-pass -o results/algorithm_breakdown.csv

# No GPU runs; prints the benchmark list and writes NA everywhere
python3 scripts/collect_algorithm_breakdown.py --dry-run -o results/preview.csv
```

### Compile overrides

```bash
# Different GPU (example: sm_75)
SM=75 ./scripts/compile_algorithm_breakdown.sh

# Full gencode string
NVCC_GENCODE="arch=compute_75,code=sm_75" ./scripts/compile_algorithm_breakdown.sh
```

The collector passes `DEBUG_MODE` automatically when it invokes the compile script; you normally do not set it yourself unless building manually.

## Benchmark matrix

Each **column** in the CSV (except the first `metric` column) is one run: a dataset file under StructX plus one XPath. Labels are `PSD[0]` … `GOOGLE_MAP[0]` as defined in `collect_algorithm_breakdown.py` (`BENCHMARK_RUNS`). To change datasets or queries, edit that list.

Invocation shape (same as `main.cu`):

```text
./output.exe <path-to-xml> <xpath-string>
```

## CSV layout

- **First row:** header. First cell is `metric`; remaining cells are the benchmark column names (e.g. `PSD[0]`, `NY_MOTOR[1]`).
- **Following rows:** one row per metric name (see table below). Each cell is the value for that metric for that column’s run.
- **`NA`:** missing or unparseable value (wrong `DEBUG_MODE` for that metric, missing XML, crashed run, etc.).

## Meaning of each metric row

| Row name (`metric` column) | Meaning |
|----------------------------|---------|
| `host_to_device_ms` | Time to copy the XML buffer from host (CPU) to device (GPU), in milliseconds. From `DEBUG_MODE=3` output. |
| `validation_ms` | Time for the UTF-8 validation stage on the GPU, in milliseconds. |
| `tokenization_ms` | Time for the tokenization stage, in milliseconds. |
| `parser_ms` | Time for the parsing / structure stage, in milliseconds. |
| `total_parsing_ms` | Sum of `host_to_device_ms` + `validation_ms` + `tokenization_ms` + `parser_ms` when all four are present; if validation is missing from the log, the sum uses the three available stage times. |
| `query_compute_ms` | Time spent executing the XPath query on the GPU, in milliseconds. |
| `query_d2h_ms` | Time to transfer query results from device to host, in milliseconds. |
| `total_query_ms` | Sum of `query_compute_ms` + `query_d2h_ms`. |
| `total_end_to_end_ms` | Sum of `total_parsing_ms` + `total_query_ms` (full pipeline time for that run, as reported by these components). |
| `tokens_count` | Number of tokens produced by tokenization for the loaded document. |
| `matches_found` | Number of XPath matches reported by the host-side match printer (integer). |
| `gpu_memory_used_mb_max` | Over all lines of the form `Used = <number> MB` in the **second** pass (`DEBUG_MODE=5`), the **largest** value, in megabytes. Summarizes peak observed GPU memory use during that run. `NA` if you used `--single-pass` or if no such lines appeared. |

All times are taken from the millisecond lines in program output (not the nanosecond duplicates).
