# Table of contents
1. [Overview](#overview)
2. [Repository Structure](#fstructure)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Preprocessing](#preprocessing)
    2. [Manual classification](#classification)
5. [Labels](#labels)
6. [Testing](#testing)

---
# 1. Overview <a name="overview"></a>

**tree-classification-tool** is a set of tools for manual species classification of trees scanned with LiDAR. It takes `.las`/`.laz` point clouds with already segmented trees (`tree_ids` field) and turns every tree into a visual report that a person can label in a desktop GUI. Key features:
- Preprocessing: splits every point cloud into single trees, renders 5 depth map views per tree (top + 4 sides, coloured with RGB when available) on CPU or CUDA, and writes a PDF report, a CSV sheet and a point cloud per tree,
- Resumable batch processing: processed files are skipped on the next run, failed files are logged and retried,
- Manual classification GUI (PySide6): one tree per page, labels chosen by code, Polish or Latin name, or by clicking a button; work resumes at the first unlabelled tree,
- Quick location check: every tree has links to Google Maps and the BDL (Bank Danych o Lasach) forest map.

The project has two independent pipelines:

```
 .las/.laz with tree_ids
          │
          ▼
 ┌──────────────────┐     {file}_trees_report.pdf    ┌──────────────────┐
 │  preprocess      │ ──► {file}_trees_report.csv ──►│  classify (GUI)  │ ──► labels in CSV
 │  (CPU / CUDA)    │     pcds/{file}_{tree_id}.laz  │                  │
 └──────────────────┘                                └──────────────────┘
```

---
# 2. Repository structure: <a name="fstructure"></a>

```
.
├── docs
│   ├── AGENTS.md                #Requirements and coding conventions
│   ├── architecture.md          #Architecture plan
│   ├── status.md                #Work log
│   └── workplan.md              #Original specification
├── src
│   └── tree_classification
│       ├── cli
│       │   ├── preprocess.py    #Preprocessing entrypoint (`preprocess` command)
│       │   └── classify.py      #Classification GUI entrypoint (`classify` command)
│       ├── core                 #Data models and protocols shared by all layers
│       ├── gui                  #PySide6 window and widgets (thin, no business logic)
│       ├── pointcloud           #LAS/LAZ reading, writing and tree extraction
│       ├── rendering
│       │   └── depth_map.py     #Depth map renderer (torch, CPU/CUDA)
│       ├── reporting            #PDF report writer/reader and CSV label store
│       ├── services             #Preprocessing and classification logic
│       └── tracking             #processed.txt / error_files.txt handling
├── tests                        #Tests, mirroring the src/tree_classification layout
│   ├── gui                      #test_app.py, test_widgets.py
│   ├── pointcloud               #test_extraction.py, test_reader.py, test_writer.py
│   ├── rendering                #test_depth_map.py
│   ├── reporting                #test_csv_store.py, test_pdf_reader.py, test_pdf_writer.py
│   ├── services                 #test_classification.py, test_preprocessing.py
│   ├── tracking                 #test_file_tracker.py
│   ├── test_classification_service.py   #End-to-end: classification pipeline
│   └── test_preprocessing_service.py    #End-to-end: preprocessing pipeline + CLI
├── labels.json                  #Label codes with Polish and Latin names
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 3. Installation: <a name="installation"></a>

The project is managed with [uv](https://docs.astral.sh/uv/). Install it first if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:
```bash
git clone https://github.com/kalmary/tree-classification-tool.git
cd tree-classification-tool

# Creates .venv, installs dependencies from uv.lock and the `preprocess` / `classify` commands
uv sync
```

Requires Python 3.12+ (uv downloads it automatically if missing).

`--device cuda` needs a CUDA-enabled build of PyTorch and an NVIDIA driver. Check it with:
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```
If it prints `False`, preprocessing still works with `--device cpu`.

---

# 4. Usage <a name="usage"></a>

All commands are run from the repository root. For every command you can use the ``--help`` flag.

## 1. Preprocessing <a name="preprocessing"></a>

Input point clouds must contain a `tree_ids` field (one integer id per point). To process files run:
```bash
uv run preprocess --input_path path/to/raw/data --output_dir path/to/output --device cuda
```
Available flags:
- ``input_path`` - single `.las`/`.laz` file or a directory with them (not searched recursively),
- ``output_dir`` - destination directory for all outputs,
- ``device`` - `cpu` (default) or `cuda`; used for depth map rendering.

For every input file `{file}` the following is created:
```
output_dir
├── processed.txt                    #Files already processed - skipped on the next run
├── error_files.txt                  #Files that failed - retried on the next run
└── {file}
    ├── pcds
    │   └── {file}_{tree_id}.laz     #Point cloud of each tree, same format as input (.las/.laz)
    ├── {file}_trees_report.pdf      #One page per tree: 5 views + lat/lon/height
    └── {file}_trees_report.csv      #One row per tree, label initialised to -2
```

PDF page `i` corresponds to CSV row `i`. The CSV has the columns:
```
tree_id,latitude,longitude,height,source_tree_id,label
```
- ``latitude`` / ``longitude`` - tree centroid in WGS84, transformed from the CRS stored in the LAS header,
- ``height`` - tree height in metres,
- ``source_tree_id`` - `{file}_{tree_id}`, also the title of the PDF page,
- ``label`` - species code from `labels.json`, `-2` means unclassified.

The command returns exit code `1` if any file failed (see `error_files.txt`) and `0` otherwise. Rerunning the same command continues where it stopped.

> **Note:** file names must contain only Latin-1 characters (no Polish letters such as `ł`, `ż`), because they are printed in the PDF with a core PDF font.

## 2. Manual classification <a name="classification"></a>

To label trees run:
```bash
uv run classify --input_path path/to/output
```
Available flags:
- ``input_path`` - directory searched recursively for `*_trees_report.csv` + `.pdf` pairs, or a single report `.pdf`/`.csv` file,
- ``labels`` - path to the labels file (default: `labels.json` in the repository root).

Reports with all trees already labelled are skipped. The others open at the first tree with label `-2`.

Working in the GUI:
- type a label **code** (e.g. `5`) or part of a **Polish/Latin name** (e.g. `dab`, `quer`) in the *Etykieta* field; case and Polish diacritics are ignored, matching labels are outlined on the list,
- or click a label button on the list,
- **Enter** or a **second click** on the selected label saves it and moves to the next tree,
- **Dalej** saves the label and goes to the next tree; after the last tree of a report it continues with the next report,
- **Poprzednie** goes back without saving changes on the current tree,
- **Zapisz i zakończ** saves the current label and closes the window,
- links at the top open the tree location in Google Maps and the BDL forest map; the progress bar shows how many trees in the report are labelled.

Labels are written to the CSV right away, so closing the window never loses saved work.

---

# 5. Labels <a name="labels"></a>

Labels are defined in `labels.json`:
```json
{
  "-2": {"latin": "Unclassified", "polish": "Nieklasyfikowane"},
  "0":  {"latin": "Pinus", "polish": "sosna"},
  "1":  {"latin": "Picea", "polish": "świerk"}
}
```
- keys are integer codes stored in the CSV `label` column,
- `-2` is reserved for unclassified trees (set by preprocessing),
- to add a species, add a new code with its Latin and Polish name and restart the GUI.

---

# 6. Testing <a name="testing"></a>

All tests live in `tests/`, in subdirectories mirroring the packages in `src/tree_classification` (e.g. `src/.../reporting/csv_store.py` → `tests/reporting/test_csv_store.py`). pytest is in the `test` dependency group, so run:
```bash
uv run --group test pytest
```
Run a single package or file:
```bash
uv run --group test pytest tests/reporting
uv run --group test pytest tests/reporting/test_csv_store.py
```
GUI tests can run without a display:
```bash
QT_QPA_PLATFORM=offscreen uv run --group test pytest
```
Tests requiring CUDA are skipped automatically when it is not available.