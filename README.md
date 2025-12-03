# 📌 yuv-feature-engineering

Feature extraction and engineering from Y luma blocks of YUV videos for video analysis and machine learning applications.
The pipeline reads a CSV with block positions, loads the corresponding frames from YUV videos, and computes an extensive set of statistical, structural, and frequency-based attributes.

---

## 📂 Key Features

- Efficient CSV processing in chunks (supports very large files).
- Direct extraction of Y (luma) blocks from YUV 4:2:0 videos.
- Calculation of statistical, structural, directional, texture and transform features.
- Support for 8-bit and 10-bit videos.
- Support for multiple video datasets/resolutions (4K, 1080p, 720p, JVET, CTC, etc.).
- Incremental writing of a new CSV with all features appended.

---

## 🧠 Extracted Features Description

Below are all features computed by the pipeline, organized by category.

---

### 1. Basic Statistics (stats_bases)

| Feature              | Description                         |
| -------------------- | ----------------------------------- |
| `blk_pixel_mean`     | Mean value of block pixels.         |
| `blk_pixel_variance` | Variance of block pixels.           |
| `blk_pixel_std_dev`  | Standard deviation.                 |
| `blk_pixel_sum`      | Sum of all pixel values in the block.|

---

### 2. Directional Statistics (stats_bases)

| Feature     | Description                                      |
| ----------- | ------------------------------------------------ |
| `blk_var_h` | Average variance per row (horizontal).           |
| `blk_var_v` | Average variance per column (vertical).          |
| `blk_std_h` | Average standard deviation per row.              |
| `blk_std_v` | Average standard deviation per column.           |

---

### 3. Contrast and Sharpness (stats_bases)

| Feature             | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `blk_min`           | Minimum pixel value in the block.                                |
| `blk_max`           | Maximum pixel value in the block.                                |
| `blk_range`         | Range (max − min).                                                |
| `blk_laplacian_var` | Variance of the Laplacian (indicator of sharpness / blur).       |

---

### 4. Complexity / Texture

| Feature       | Description                                                           |
| ------------- | --------------------------------------------------------------------- |
| `blk_entropy` | Shannon entropy of the block (texture complexity).                    |

---

### 5. Sobel Gradients (grad_bases_sobel)

| Feature                | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `blk_sobel_gv`         | Vertical gradient (horizontal edges).            |
| `blk_sobel_gh`         | Horizontal gradient (vertical edges).            |
| `blk_sobel_mag`        | Mean gradient magnitude.                         |
| `blk_sobel_dir`        | Mean gradient direction (in degrees).            |
| `blk_sobel_razao_grad` | Ratio gh / gv.                                   |

---

### 6. Prewitt Gradients (grad_bases_prewitt)

| Feature                  | Description                                   |
| ------------------------ | --------------------------------------------- |
| `blk_prewitt_gv`         | Vertical gradient via Prewitt.                |
| `blk_prewitt_gh`         | Horizontal gradient via Prewitt.              |
| `blk_prewitt_mag`        | Mean magnitude via Prewitt.                   |
| `blk_prewitt_dir`        | Mean direction via Prewitt.                   |
| `blk_prewitt_razao_grad` | Ratio gh / gv (Prewitt).                      |

---

### 7. Hadamard Transform (hadamard_bases)

| Feature                | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| `blk_had_dc`           | DC coefficient (overall brightness).                              |
| `blk_had_energy_total` | Sum of squares of all coefficients.                               |
| `blk_had_energy_ac`    | AC energy (total − DC²).                                           |
| `blk_had_max`          | Largest absolute coefficient.                                      |
| `blk_had_min`          | Smallest absolute coefficient.                                     |
| `blk_had_topleft`      | Coefficient H[0,0] (DC).                                           |
| `blk_had_topright`     | Top-right corner coefficient.                                      |
| `blk_had_bottomleft`   | Bottom-left corner coefficient.                                    |
| `blk_had_bottomright`  | Bottom-right corner coefficient.                                   |

---

## 📦 Project Structure

```python
project/
├── config.py         # Paths, CSV separators, global settings
├── features.py       # Implementation of feature extraction functions
├── process_yuv.py    # Main routine: read CSV + YUV processing
├── README.md
└── data/
```

---

## ⚙️ Example Configuration (`config.py`)

```python
# Project directory
PROJECT_FOLDER = '/home/carolinesc/mestrado'

# Input and output CSV files
csv_input_file = os.path.join(PROJECT_FOLDER, "features.csv")
csv_output_file = os.path.join(PROJECT_FOLDER, "new-features.csv")

CSV_READ_SEP = ','
CSV_WRITE_SEP = ';'
CHUNK_SIZE = 90_000_000

# Required CSV columns
COL_FRAME = 'frame'
COL_X = 'x'
COL_Y = 'y'
COL_WIDTH = 'Width'
COL_HEIGHT = 'Height'
COL_FRAMEWIDTH = 'FrameWidth'
COL_FRAMEHEIGHT = 'FrameHeight'
COL_BITDEPTH = 'BitDepth'
```

▶️ Run
```bash
python process_yuv.py
```
The new CSV will be generated as: `new-features.csv`

---

## 🛠️ C++ Implementation

In addition to the Python version, there is an option with the same feature extraction routines implemented in C++ — useful for direct integration into encoder code.

What the C++ implementation provides
- Equivalent functions to those in `features.py`: mean, variance, gradients (Sobel/Prewitt), entropy, Laplacian variance and the Hadamard transform.
