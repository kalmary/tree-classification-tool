# intro

1. this repository is an app for manual pcd (point cloud) classification
2. each point cloud is a separate tree lidar scan
3. program consists of two main pipelines:
  3.1 data preprocessing
  3.2 manual classification


# data preprocessing

1. .laz/ .laz files are handled
 - program takes input path (cli argument) of single file to process. if path is a directory, all files in the directory are processed.
 - processed files are saved into processed.txt file (output dir, hardcoded, next to src). they will not be processed in next program run, if they were processed before.
 - file_path which processing caused an error is saved into error_files.txt file (output dir, hardcoded, next to src). its just information, we try to process this files in next run: no omitting this.
2. files should have tree_ids field
3. each tree pcd is found as:
```python
for tree_id in numpy.unique(tree_ids):
    tree_pcd = points[tree_ids == tree_id]
```
4. tree is transformed into 5 images:
  - 4 of them represent side views, one top view
  - it is ran on cpu, or cuda (cli argument)
  ```python
  def cloud2sideView(points: torch.Tensor,
                       resolution_xy: int | None = None,
                       margin_ratio: float = 0.05) -> torch.Tensor:
 
        points = points.type(torch.float64)

        min_xyz = points.min(dim=0).values
        max_xyz = points.max(dim=0).values

        center = (min_xyz + max_xyz) / 2
        max_range = (max_xyz - min_xyz).max()
        cube_half = max_range / 2 * (1 + 2 * margin_ratio)

        cube_min = center - cube_half
        cube_max = center + cube_half

        def to_grid(val, min_val, max_val):
            return torch.clamp(
                ((val - min_val) / (max_val - min_val + 1e-8) * (resolution_xy - 1)).long(),
                0, resolution_xy - 1
            )

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        gx = to_grid(x, cube_min[0], cube_max[0])
        gy = to_grid(y, cube_min[1], cube_max[1])
        gz = to_grid(z, cube_min[2], cube_max[2])

        views = []

        def build_depth_map(indices_2d, distances, flip_y=False, flip_x=False):
            y_idx, x_idx = indices_2d
            if flip_y:
                y_idx = resolution_xy - 1 - y_idx
            if flip_x:
                x_idx = resolution_xy - 1 - x_idx

            flat_indices = y_idx * resolution_xy + x_idx
            depth_map = torch.full((resolution_xy * resolution_xy,), float('inf'),
                                   dtype=torch.float64, device=distances.device)
            depth_map = torch.scatter_reduce(depth_map, 0, flat_indices, distances,
                                             reduce='amin', include_self=True)

            img = depth_map.view(resolution_xy, resolution_xy)
            valid_mask = torch.isfinite(img)

            if torch.any(valid_mask):
                values = img[valid_mask]
                min_val = values.min()
                max_val = values.max()
                normalised = (max_val - values) / (max_val - min_val + 1e-8)
                normalised = normalised * (1.0 - 1.0 / 255.0) + (1.0 / 255.0)
                img = img.clone()
                img[valid_mask] = normalised
                img[~valid_mask] = 0.0
            else:
                img = torch.zeros_like(img)

            return img.type(torch.float32)

        dist_top = cube_max[2] - z
        views.append(build_depth_map((gy, gx), dist_top))

        dist_front = cube_max[1] - y
        views.append(build_depth_map((gz, gx), dist_front, flip_y=True))

        dist_back = y - cube_min[1]
        views.append(build_depth_map((gz, gx), dist_back, flip_y=True, flip_x=True))

        dist_left = cube_max[0] - x
        views.append(build_depth_map((gz, gy), dist_left, flip_y=True))

        dist_right = x - cube_min[0]
        views.append(build_depth_map((gz, gy), dist_right, flip_y=True, flip_x=True))

        return torch.stack(views, dim=0).type(torch.float32)
  ```
  Above implementation generates the depth maps for the 5 camera views. rgb N,3 optional array should be added as an input too: if given, the rgb values are used to colour the depth maps instead.
  
  5. images are 256 x 256 resolution. each image is put onto a pdf page:
    - 1st row: 3 imgs
    - 2nd row: 2 imgs + additional info:
      - lat + longt position of a tree,
      - height of a tree
      - original_filename_tree_id - text formatted like this with proper data included
  6. all images/ pdf pages are saved into single pdf file: original_laz_filename_trees_report.pdf - change text to proper data. one laz file with mulitple trees -> one file with, multiple pdf pages
  7. alongside pdf report, csv sheet is created and appended with tree data for each pdf page:
   - tree_id, lat, longt, height, original_filename_tree_id, label (last field set as -2)
   - one laz file with multiple trees -> one csv sheet with multiple rows
  8. alongside pdf report, separate tree pcds are saved as .laz/ .las (Same format as the original file). fields should be:
    - xyz points
    - rgb points (only if available)
  9. all data is saved in directory goal/original_laz_filename/. point clouds with trees are in subdirectory pcds/. goal is dirctory defined by user (cli argument)
  10. program is ran from cli. arguments are marked as cli arguments.



  # manual classification
  this is a separate part ran completely independently of the 1st part.
  1. pdf site by site is opened,
  2. for each side label value must be given and saved to csv file (-2 value till now). it must be int
  3. GUI is required. Single tab for a pdf side:
    - buttons: next, previous, save and quit
    - on the upper side of the tab pdf site is displayed.
  - on the lower side of the tab there are buttons.
  - next to buttons empty field for label input:
    - user can use integer value,
    - user can write text in latin,
    - user can write text in polish
    - user can use lista rozwijana - all in same field.
  - once next button is clicked label is written to csv file
  - once back button is clicked label is not saved and user is taken to previous pdf site: label can be choosen again
  - once save and quit button is clicked label is saved and program exits

  4. non processed tree is labeled as -2. once program is stopped and ran again it should start off from the very first -2 label.
  5. program to run requires cli:
    - directory_path: path to directory containing pdf and csv files. program iterates through all files in the directory. if path is file, program processes only that file with matching csv file.
# Implementation Plan

Based on the architecture and agent guidelines, here is the 7-phase implementation plan to build the entire solution systematically.

### Phase 1: Core Foundation & Tracking
1. Define the `Core` data models (`PointCloud`, `TreeData`, `TreeLabel`, `TreeReport`).
2. Define the `Protocol` interfaces (`PointCloudReader`, `PointCloudWriter`, `Renderer`, `ReportWriter`, `LabelStore`).
3. Implement `tracking/file_tracker.py` to handle `processed.txt` and `error_files.txt`.

### Phase 2: Pointcloud Domain (LiDAR processing)
1. Implement `laspy`-based `PointCloudReader` and `PointCloudWriter`.
2. Implement tree extraction logic to split a parent `PointCloud` into individual `TreeData` objects.
3. Compute bounding box, height, and project centroid to WGS84 (Lat/Long) using `pyproj`.

### Phase 3: Rendering Domain
1. Implement `Renderer` in `rendering/depth_map.py` utilizing the `torch` `cloud2sideView` algorithm.
2. Extend the algorithm to support RGB colorization if the input point cloud contains RGB channels.

### Phase 4: Reporting & Persistence Domain
1. Implement `pdf_writer.py` using `fpdf2` to lay out the 5 views + metadata on a single A4 page.
2. Implement `csv_store.py` to act as the `LabelStore` handling CSV persistence (initializing new trees with `-2`).

### Phase 5: Preprocessing Service & CLI
1. Build `PreprocessingService` that orchestrates Phases 1-4 into the full data preprocessing pipeline.
2. Build the `tree-preprocess` CLI entry point using `argparse`.
3. Verify the preprocessing pipeline with integration tests.

### Phase 6: Classification Service & PDF Reading
1. Implement `pdf_reader.py` using `PyMuPDF` (`fitz`) to extract page images from reports.
2. Build `ClassificationService` to load `labels.json`, manage navigation state, read PDF pages, and sync changes to `LabelStore`.

### Phase 7: GUI Presentation Layer & Final CLI
1. Build `PySide6` custom widgets (`LabelCombobox`, `PdfPageViewer`).
2. Build the main `ClassificationGUI` window that consumes `ClassificationService`.
3. Create the `tree-classify` CLI entry point.
4. Perform end-to-end testing of the manual classification pipeline.
