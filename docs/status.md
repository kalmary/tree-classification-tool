# Status

## Work Log

- Defined the `# TASK` section in `docs/AGENTS.md` based on `docs/workplan.md`: two pipelines (data preprocessing + manual classification GUI) with inputs, outputs, processing steps, and resilience requirements.
- Implemented Phase 1: Core Foundation & Tracking. Created data models in `core/models.py`, abstract protocols in `core/protocols.py`, and `tracking/file_tracker.py` with inline tests. All tests passed.
- Implemented Phase 2: Pointcloud Domain. Created `LaspyReader`, `LaspyWriter`, and tree `extraction` logic with coordinate transformation using `pyproj`. Unit tests passed successfully.
- Configured pytest in `pyproject.toml` to automatically discover inline tests within standard `.py` files across the `src/` directory.
- Implemented Phase 3: Rendering Domain. Added `torch` dependency and `DepthMapRenderer` in `rendering/depth_map.py` producing the 5 `cloud2sideView` views on CPU or CUDA, with optional RGB colouring from the nearest point. Inline tests cover resolution, depth range, RGB colouring, and view mirroring; all passed.
- Implemented CsvLabelStore in reporting/csv_store.py (Phase 4, CSV part): spec header tree_id,lat,long,height,original_filename_tree_id,label, overwrite-only atomic writes, strict header/row validation, index-based update_label. 9 inline tests passed.
- Fixed CSV column names (documentation error): header is now tree_id,latitude,longitude,height,source_tree_id,label, identical to TreeLabel fields. Updated csv_store.py, AGENTS.md, architecture.md, workplan.md; added test_columns_match_tree_label_fields. 10 tests passed.
- Implemented PdfReportWriter in reporting/pdf_writer.py (Phase 4, PDF part): one A4 landscape page per tree (title, 3+2 view tiles with captions, lat/lon/height), view validation, atomic write, core Helvetica font (Latin-1 only, non-Latin-1 text fails loudly). Dependencies: fpdf2 (main, PDF generation), pillow (main, imported directly for view-to-image conversion), pymupdf (test, reads generated PDFs in tests; moves to main in Phase 6). 9 inline tests passed.