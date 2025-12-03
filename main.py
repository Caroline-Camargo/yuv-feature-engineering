import os
import cv2
import numpy as np
import pandas as pd
import yuvio

from config import *
from features import *


def process_csv_with_yuv(
    csv_input,
    csv_output,
    video_paths_mapping,
    cols_to_check=COLS_TO_CHECK,
    chunk_size=CHUNK_SIZE,
    csv_read_sep=CSV_READ_SEP,
    csv_write_sep=CSV_WRITE_SEP,
    print_block_pixels=PRINT_BLOCK_PIXELS,
):
    """Process the CSV and add block-level features extracted from YUV blocks."""

    print(f"Reading: {csv_input} in chunk mode...")
    first_chunk = True
    total_lines_processed = 0

    # --- Definition of feature columns (block-level) ---
    stats_bases = [
        'blk_pixel_mean', 'blk_pixel_variance', 'blk_pixel_std_dev', 'blk_pixel_sum',
        'blk_var_h', 'blk_var_v', 'blk_std_v', 'blk_std_h',
        'blk_min', 'blk_max', 'blk_range',
        'blk_laplacian_var',
        'blk_entropy'
    ]
    grad_bases_sobel = [
        'blk_sobel_gv', 'blk_sobel_gh', 'blk_sobel_mag', 'blk_sobel_dir', 'blk_sobel_razao_grad'
    ]
    grad_bases_prewitt = [
        'blk_prewitt_gv', 'blk_prewitt_gh', 'blk_prewitt_mag', 'blk_prewitt_dir', 'blk_prewitt_razao_grad'
    ]
    hadamard_bases = [
        'blk_had_dc', 'blk_had_energy_total', 'blk_had_energy_ac',
        'blk_had_max', 'blk_had_min',
        'blk_had_topleft', 'blk_had_topright', 'blk_had_bottomleft', 'blk_had_bottomright'
    ]

    base_features = stats_bases + grad_bases_sobel + grad_bases_prewitt

    if print_block_pixels:
        feature_columns = base_features + hadamard_bases + ['blk_pixels']
    else:
        feature_columns = base_features + hadamard_bases

    # -----------------------------------------------------------------

    for chunk in pd.read_csv(csv_input, sep=csv_read_sep, chunksize=chunk_size):
        print(f"\nProcessing chunk with {len(chunk)} rows (Total processed: {total_lines_processed})...")

        # Data cleaning
        for c in cols_to_check:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
            else:
                chunk[c] = np.nan

        rows_with_nan = chunk[chunk[cols_to_check].isna().any(axis=1)]
        if not rows_with_nan.empty:
            for idx, row in rows_with_nan.iterrows():
                nan_cols = row[cols_to_check].isna()
                print(f"Row {idx} removed due to NaN in columns: {list(nan_cols[nan_cols].index)}")
                print(row)
       
        # Drop rows missing the required columns
        chunk.dropna(subset=cols_to_check, inplace=True)
       
        # Force integer type for main columns
        chunk[cols_to_check] = chunk[cols_to_check].astype(int)

        # Initialize columns
        for col in feature_columns:
            chunk[col] = np.nan

            # Group by video and frame
            grouped = chunk.groupby([COL_VIDEO, COL_FRAME])

        for (video_name, frame_num), group in grouped:
            video_path = video_paths_mapping.get(video_name)
            if not video_path or not os.path.exists(video_path):
                print(f"   ERROR: video '{video_name}' not found at {video_path}.")
                continue

            ref = group.iloc[0]

            try:
                width = ref[COL_FRAMEWIDTH]
                height = ref[COL_FRAMEHEIGHT]
                bit_depth = ref[COL_BITDEPTH]
                if bit_depth > 8:
                    pix_fmt = 'yuv420p10le'
                else:
                    pix_fmt = 'yuv420p'
                Y, U, V = yuvio.imread(video_path, width, height, pix_fmt, frame_num)
            except Exception as e:
                csv_lines = group.index.tolist()
                print(f"[Video: {video_name} | Frame: {frame_num} | CSV rows: {csv_lines}] ERROR reading frame with yuvio: {e}")
                continue

            # Iterate over the lines (blocks) of the CSV file.
            for i, row in group.iterrows():
                blk_orig = extract_block(Y, row[COL_X], row[COL_Y], row[COL_WIDTH], row[COL_HEIGHT])
                if blk_orig is None:
                    print(f"   ERROR: block not extracted for row {i} — skipping.")
                    continue

                blk_float = blk_orig.astype(np.float32)

                if print_block_pixels:
                    max_elems = 200
                    try:
                        if blk_float.size <= max_elems:
                            blk_str = np.array2string(blk_float, precision=2, separator=', ', max_line_width=200)
                            print(f"  Block (Video={video_name} Frame={frame_num} X={row['x']} Y={row['y']} size={blk_float.shape}):")
                            print(blk_str)
                        else:
                            blk_str = np.array2string(blk_float[:5, :5], precision=2, separator=', ', max_line_width=200)
                            blk_str = f"[TRUNCATED: only top-left 5x5 shown] {blk_str}"
                            print(f"  Large block (Video={video_name} Frame={frame_num} X={row['x']} Y={row['y']} size={blk_float.shape}, elements={blk_float.size}) - showing top-left 5x5:")
                            print(blk_str)
                        chunk.at[i, 'blk_pixels'] = blk_str
                    except Exception as e:
                        print(f"   ERROR converting block to string at row {i}: {e}")
                        chunk.at[i, 'blk_pixels'] = '[ERROR]'

                # --- Start of calculations (block-level) ---
                try:
                    mean_val, var_val, std_val, sum_val = calculate_basic_features_cv(blk_float)
                    chunk.at[i, 'blk_pixel_mean'] = mean_val
                    chunk.at[i, 'blk_pixel_variance'] = var_val
                    chunk.at[i, 'blk_pixel_std_dev'] = std_val
                    chunk.at[i, 'blk_pixel_sum'] = sum_val

                    vH, vV, dV, dH = calculate_stats_cv(blk_float)
                    chunk.at[i, 'blk_var_h'] = vH
                    chunk.at[i, 'blk_var_v'] = vV
                    chunk.at[i, 'blk_std_v'] = dV
                    chunk.at[i, 'blk_std_h'] = dH

                    mGv_s, mGh_s, mag_s, dir_s, razao_s = calculate_gradients_sobel_cv(blk_float)
                    chunk.at[i, 'blk_sobel_gv'] = mGv_s
                    chunk.at[i, 'blk_sobel_gh'] = mGh_s
                    chunk.at[i, 'blk_sobel_mag'] = mag_s
                    chunk.at[i, 'blk_sobel_dir'] = dir_s
                    chunk.at[i, 'blk_sobel_razao_grad'] = razao_s

                    mGv_p, mGh_p, mag_p, dir_p, razao_p = calculate_gradients_prewitt_cv(blk_float)
                    chunk.at[i, 'blk_prewitt_gv'] = mGv_p
                    chunk.at[i, 'blk_prewitt_gh'] = mGh_p
                    chunk.at[i, 'blk_prewitt_mag'] = mag_p
                    chunk.at[i, 'blk_prewitt_dir'] = dir_p
                    chunk.at[i, 'blk_prewitt_razao_grad'] = razao_p

                    b_min, b_max, b_range = calculate_contrast_features_cv(blk_orig)
                    chunk.at[i, 'blk_min'] = b_min
                    chunk.at[i, 'blk_max'] = b_max
                    chunk.at[i, 'blk_range'] = b_range

                    chunk.at[i, 'blk_laplacian_var'] = calculate_laplacian_cv(blk_float)

                    chunk.at[i, 'blk_entropy'] = calculate_entropy_cv(blk_orig)

                    dc, energy_total, energy_ac, max_coef, min_coef, tl, tr, bl, br = calculate_hadamard_block_features(blk_orig)
                    chunk.at[i, 'blk_had_dc'] = dc
                    chunk.at[i, 'blk_had_energy_total'] = energy_total
                    chunk.at[i, 'blk_had_energy_ac'] = energy_ac
                    chunk.at[i, 'blk_had_max'] = max_coef
                    chunk.at[i, 'blk_had_min'] = min_coef
                    chunk.at[i, 'blk_had_topleft'] = tl
                    chunk.at[i, 'blk_had_topright'] = tr
                    chunk.at[i, 'blk_had_bottomleft'] = bl
                    chunk.at[i, 'blk_had_bottomright'] = br
                except cv2.error as e:
                    print(f"   WARNING: OpenCV error while processing block at row {i} (Shape: {blk_orig.shape}). Possibly block smaller than 3x3. {e}. Filling with NaN.")
                    continue
                except Exception as e:
                    print(f"   ERROR: Unexpected error while processing block at row {i} (Shape: {blk_orig.shape}): {e}. Filling with NaN.")
                    continue

        # Save processed chunk
        print(f"Saving chunk to disk...")
        chunk.to_csv(csv_output, mode='w' if first_chunk else 'a', header=first_chunk, index=False, sep=csv_write_sep)
        first_chunk = False
        total_lines_processed += len(chunk)

    print(f"\n Processing completed. Final file saved at: {csv_output}")


if __name__ == "__main__":
    process_csv_with_yuv(
        csv_input_file,
        csv_output_file,
        video_paths_yuv,
        cols_to_check=COLS_TO_CHECK,
        chunk_size=CHUNK_SIZE,
        csv_read_sep=CSV_READ_SEP,
        csv_write_sep=CSV_WRITE_SEP,
        print_block_pixels=PRINT_BLOCK_PIXELS,
    )
