<!-- # StructX -->

<p align="center">
  <img src="assets/logo.png" alt="StructX logo" width="320">
</p>

<!-- <p align="center">
  <a href="https://github.com/ashkanvg/StructX">GitHub repository</a>
</p> -->

StructX is a GPU-accelerated pipeline that loads XML, validates and tokenizes it on the device (GPU), builds a structural representation, and evaluates a subset of XPath-style queries. The main program is a single CUDA translation unit rooted at [`src/main.cu`](src/main.cu): `nvcc` pulls in the other `.cu` / `.cpp` sources via `#include`, so you compile one entry file and produce one binary (for example `output.exe`).

<!-- This README focuses on the **`src/`** tree. Other checkouts may include sibling variants (for example sorting or hash-map experiments); the XML debug-pack scripts still reference **`src_sort/main.cu`** in places—see [Scripts](#scripts). -->

## Hardware and software requirements

- **GPU:** NVIDIA GPU with a CUDA compute capability that matches your `nvcc` `-gencode` flags (see below).
- **CUDA toolkit:** `nvcc` on your `PATH` (often `/usr/local/cuda/bin/nvcc`). A recent toolkit version compatible with your driver is recommended.
- **Host compiler:** A C++ compiler supported by your CUDA version (GCC/Clang per NVIDIA’s compatibility matrix).
- **Python 3:** Optional but required for the benchmark and XML test scripts under [`scripts/`](scripts/).

On some lab machines (e.g. Red Hat with an older default GCC), you may need a newer toolchain before CUDA will compile cleanly.
<!-- , for example: -->
<!-- 
```bash
scl enable gcc-toolset-13 bash
``` -->

## Datasets

Download the shared datasets from Google Drive and place them under the StructX project so paths such as `./dataset/psd7003.xml` resolve correctly:

- [StructX datasets (Google Drive)](https://drive.google.com/drive/folders/12x8yM3wFoDTTcZp80o4i7CH0SYvYc3Yy?usp=sharing)

The archive includes folders like `data_gov`, `json2xml`, and `xml_debug_pack`, plus large XML files (e.g. `psd7003.xml`, `lab_samples_1gb.xml`, `SwissProt.xml`). By default the `dataset/` directory in this repo may be empty until you copy or unpack files there.


---

## How to build and run (match count and sample matches)

Use **`DEBUG_MODE=0`** for a normal run with minimal diagnostic noise: you get the total match count and up to **10** printed matches.

### 1. Install datasets

Download from the Google Drive link above and put the files under the StructX root (e.g. `dataset/psd7003.xml`, `dataset/data_gov/…`, etc.).

### 2. Compile

From the **StructX repository root**:

```bash
/usr/local/cuda/bin/nvcc -DDEBUG_MODE=0 -O3 -o output.exe ./src/main.cu -w -gencode=arch=compute_61,code=sm_61
```

If `nvcc` is already on your `PATH`, you can use `nvcc` instead of the full path. 

**Important-1:** Adjust `compute_XX,code=sm_XX` for your GPU. Ensure **CUDA architecture flags match your GPU**. Replace `61` with your device’s compute capability (see [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)):

```bash
-gencode=arch=compute_61,code=sm_61   # example: Pascal-class GPU
```

If the flags do not match your hardware, the binary may fail to run or run incorrectly.


**Important-2:** Always pass **`-DDEBUG_MODE=0`** for this “production” style run. If you omit `-DDEBUG_MODE`, the code defaults to verbose debug mode (`DEBUG_MODE=1`) in [`src/utils.cu`](src/utils.cu).

### 3. General invocation shape:

```text
./output.exe <path-to-xml> <xpath-query>
```


### 4. Run a first example

Pass the XML path and the XPath string as arguments (recommended so you do not have to edit [`src/main.cu`](src/main.cu)):

```bash
./output.exe ./dataset/psd7003.xml /ProteinDatabase/ProteinEntry/protein/name
```

### 5. What you should see

- A line of the form **`Matches found:`** *n* with the total number of matches.
- Up to **10** lines **`Match 0:`** … **`Match 9:`** with excerpted text for each match (the implementation prints at most 10 matches).



---

## How to build and run (stage timings)

For **coarse, stage-level timings** in milliseconds (validation, tokenization, structure recognition, query, etc.), build with **`DEBUG_MODE=3`**:

```bash
nvcc -DDEBUG_MODE=3 -O3 -o output.exe ./src/main.cu -w -gencode=arch=compute_61,code=sm_61
```

Run the same way as above. The program prints lines such as `⏱️ … execution time: … ms` for each major stage. Use this mode when profiling the pipeline rather than when you only care about correctness and match output.

---

## `DEBUG_MODE` reference

The numeric mode is selected at **compile time** with `-DDEBUG_MODE=N`.

| Mode | Purpose |
|------|--------|
| **0** | **Production:** minimal extra logging; suitable for timing end-to-end behavior without verbose internals (and for clean match listing). |
| **1** | **Verbose correctness:** detailed prints through validation, tokenization, parsing, and query (including optional tag-condition substeps; modes **6–10** narrow which tag-condition debug prints appear—mode **1** effectively enables the full set in that path). |
| **2** | **Fine-grained timing:** many per-kernel or per-step timings (nanoseconds/milliseconds) beyond the coarse stages. |
| **3** | **Coarse stage-level timing:** one timing block per major pipeline stage (host-to-device, validation, tokenization, parser, query, etc.). |
| **4** | **Lightweight summary stats:** e.g. character counts / compact summaries to stderr/stdout depending on stage. |
| **5** | **GPU memory usage:** prints `GPU Memory Usage: Used = … MB` style lines during execution (used by the algorithm-breakdown collector for peak memory). |

<!-- Modes **6–10** (tag-condition query path only) are finer-grained toggles for debugging specific steps in that branch; see comments in [`src/stages/query.cu`](src/stages/query.cu). -->

---

## Scripts

More detail lives in [`scripts/README.md`](scripts/README.md). Summary:

### Algorithm breakdown (CSV benchmarks)

Build helper (defaults to `DEBUG_MODE=3`, adjustable `SM` / `NVCC_GENCODE`):

```bash
./scripts/compile_algorithm_breakdown.sh
```

Run the full benchmark matrix and write a wide CSV (default two-pass: timings plus peak GPU memory when possible):

```bash
./scripts/run_algorithm_breakdown.sh -o results/algorithm_breakdown.csv
```

This compiles and runs against [`src/main.cu`](src/main.cu) via the compile script.

### XML debug pack runners

Scripts such as [`s./cripts/run_valid_xml_debug_pack.sh`](scripts/run_valid_xml_debug_pack_cdata.sh), and [`scripts/run_invalid_xml_debug_pack.sh`](scripts/run_invalid_xml_debug_pack.sh),  compile **`./src/main.cu`** and drive Python runners over the `xml_debug_pack` corpus.

<!-- **Note:** The primary documented entry point for this repository is [`src/main.cu`](src/main.cu). If your checkout does not include a `src_sort/` tree, these shell scripts will fail until you add that variant or change the compile line in those scripts to point at `./src/main.cu` instead. -->

---

## Optional: CUDA `PATH` and libraries

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

<!-- ## Optional: debugging and sanitizers

- Debug build (host/device): e.g. `nvcc -g -G -o output.exe ./src/main.cu -w -gencode=...`
- [CUDA Compute Sanitizer](https://docs.nvidia.com/cuda/compute-sanitizer/index.html): `/usr/local/cuda/bin/compute-sanitizer ./output.exe ...` -->

## Reference machine (example)

One known configuration used during development: GCC 11.x, CUDA 12.1, NVIDIA Quadro P4000 (`sm_61`), Intel Xeon E3-1225 v6. Your settings should follow your own GPU and toolkit versions.
