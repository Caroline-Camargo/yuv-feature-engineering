import cv2
import numpy as np
import pandas as pd
import os
import yuvio

# ======================================================================
# DESCRIÇÃO DAS FEATURES CALCULADAS
# ======================================================================

'''
--- 1. Estatísticas Básicas (stats_bases) ---
'blk_pixel_mean': Média aritmética de todos os pixels do bloco.
'blk_pixel_variance': Variância de todos os pixels do bloco.
'blk_pixel_std_dev': Desvio padrão de todos os pixels do bloco.
'blk_pixel_sum': Soma de todos os valores de pixel do bloco.

--- 2. Estatísticas Direcionais (stats_bases) ---
'blk_var_h': Média da variância de cada linha (horizontal).
'blk_var_v': Média da variância de cada coluna (vertical).
'blk_std_h': Média do desvio padrão de cada linha (horizontal).
'blk_std_v': Média do desvio padrão de cada coluna (vertical).

--- 3. Contraste e Nitidez (stats_bases) ---
'blk_min': Valor mínimo de pixel no bloco.
'blk_max': Valor máximo de pixel no bloco.
'blk_range': Diferença entre o máximo e mínimo (max - min).
'blk_laplacian_var': Variância do Laplaciano. Mede a nitidez/blur.
                     Valores altos indicam bordas nítidas; baixos indicam blur.

--- 4. Complexidade (stats_bases) ---
'blk_entropy': Entropia de Shannon. Mede a complexidade da textura.
               Valores altos indicam textura complexa/ruído; baixos indicam áreas suaves.

--- 5. Gradientes (Sobel) (grad_bases_sobel) ---
'blk_sobel_gv': Média do gradiente vertical (detecta bordas horizontais).
'blk_sobel_gh': Média do gradiente horizontal (detecta bordas verticais).
'blk_sobel_mag': Média da magnitude do gradiente (força total das bordas).
'blk_sobel_dir': Média da direção (ângulo) do gradiente.
'blk_sobel_razao_grad': Razão entre gradiente horizontal e vertical (gh / gv).

--- 6. Gradientes (Prewitt) (grad_bases_prewitt) ---
'blk_prewitt_gv': Similar ao Sobel, mas com kernel Prewitt (média do gradiente vertical).
'blk_prewitt_gh': Similar ao Sobel, mas com kernel Prewitt (média do gradiente horizontal).
'blk_prewitt_mag': Média da magnitude do gradiente usando Prewitt.
'blk_prewitt_dir': Média da direção do gradiente usando Prewitt.
'blk_prewitt_razao_grad': Razão entre gradiente horizontal e vertical (Prewitt).

--- 7. Transformada de Hadamard (hadamard_bases) ---
'blk_had_dc': Coeficiente DC — valor médio do bloco na frequência zero (representa o brilho geral).
'blk_had_energy_total': Energia total — soma dos quadrados de todos os coeficientes (intensidade global do bloco).
'blk_had_energy_ac': Energia AC — soma dos quadrados dos coeficientes não-DC (representa variação/estrutura do bloco).
'blk_had_max': Máximo coeficiente — maior valor absoluto entre todos os coeficientes Hadamard.
'blk_had_min': Mínimo coeficiente — menor valor absoluto entre todos os coeficientes Hadamard.
'blk_had_topleft': Coeficiente no canto superior esquerdo (H[0,0], mesmo que DC) — captura padrão base/nível DC.
'blk_had_topright': Coeficiente no canto superior direito (H[0,-1]) — captura padrões de borda/localização.
'blk_had_bottomleft': Coeficiente no canto inferior esquerdo (H[-1,0]) — captura padrões de borda/localização.
'blk_had_bottomright': Coeficiente no canto inferior direito (H[-1,-1]) — captura padrões de borda/localização.
'''

# ======================================================================


# Flag para ativar/desativar o print dos pixels do bloco
PRINT_BLOCK_PIXELS = False

# Configurações de Diretórios
#PROJECT_FOLDER = '/home/yasmin/Videos/ctc-vvc'
PROJECT_FOLDER = '/home/carolinesc/mestrado'

#YUV_BASE_FOLDER = PROJECT_FOLDER
YUV_BASE_FOLDER = '/data/videos/'

YUV_VIDEO_FOLDERS = {
    '4k': os.path.join(YUV_BASE_FOLDER, '4k'),
    '4k_jvet': os.path.join(YUV_BASE_FOLDER, '4k_jvet'),
    '1080p': os.path.join(YUV_BASE_FOLDER, '1080p'),
    '720p': os.path.join(YUV_BASE_FOLDER, '720p'),
}


# ==============================
# Mapeamento de vídeos YUV
# ==============================
video_paths_yuv = {
    'Vidyo3_1280x720_60': os.path.join(YUV_VIDEO_FOLDERS['720p'], 'Vidyo3_1280x720_60.yuv'),
    'Vidyo4_1280x720_60': os.path.join(YUV_VIDEO_FOLDERS['720p'], 'Vidyo4_1280x720_60.yuv'),
    'YachtRide_1920x1080_120fps_420_8bit_YUV': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'YachtRide_1920x1080_120fps_420_8bit_YUV.yuv'),
    'CrowdRun_1920x1080_25': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'CrowdRun_1920x1080_25.yuv'),
    'ParkScene_1920x1080_24': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'ParkScene_1920x1080_24.yuv'),
    'Beauty_3840x2160_120fps_420_10bit_YUV': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'Beauty_3840x2160_120fps_420_10bit_YUV.yuv'),
    'RollerCoaster_4096x2160_60fps_10bit_420_jvet': os.path.join(YUV_VIDEO_FOLDERS['4k_jvet'], 'RollerCoaster_4096x2160_60fps_10bit_420_jvet.yuv'),
    'ToddlerFountain_4096x2160_60fps_10bit_420_jvet': os.path.join(YUV_VIDEO_FOLDERS['4k_jvet'], 'ToddlerFountain_4096x2160_60fps_10bit_420_jvet.yuv'),

    # CTC VIDEOS
    # # --- 240p (CLASS D) ---
    # 'BQSquare': os.path.join(YUV_VIDEO_FOLDERS['240p'], 'BQSquare_416x240_60.yuv'),
    # 'BlowingBubbles': os.path.join(YUV_VIDEO_FOLDERS['240p'], 'BlowingBubbles_416x240_50.yuv'),
    # 'BasketballPass': os.path.join(YUV_VIDEO_FOLDERS['240p'], 'BasketballPass_416x240_50.yuv'),
    # 'RaceHorses': os.path.join(YUV_VIDEO_FOLDERS['240p'], 'RaceHorses_416x240_30.yuv'),

    # # --- 480p (CLASS C) ---
    # 'BQMall': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'BQMall_832x480_60.yuv'),
    # 'BasketballDrill': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'BasketballDrill_832x480_50.yuv'),
    # 'RaceHorsesC': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'RaceHorsesC_832x480_30.yuv'),
    # 'PartyScene': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'PartyScene_832x480_50.yuv'),
    # 'BasketballDrillText': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'BasketballDrillText_832x480_50.yuv'),

    # # --- 720p (CLASS E) ---
    # 'FourPeople': os.path.join(YUV_BASE_FOLDER, 'FourPeople_1280x720_60.yuv'),
    # 'Johnny': os.path.join(YUV_BASE_FOLDER, 'Johnny_1280x720_60.yuv'),
    # 'KristenAndSara': os.path.join(YUV_BASE_FOLDER, 'KristenAndSara_1280x720_60.yuv'),
    # 'SlideEditing': os.path.join(YUV_BASE_FOLDER, 'SlideEditing_1280x720_30.yuv'),
    # 'SlideShow': os.path.join(YUV_BASE_FOLDER, 'SlideShow_1280x720_20.yuv'),

    # # --- 1080p (CLASS B) ---
    # 'BQTerrace': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'BQTerrace_1920x1080_60.yuv'),
    # 'Cactus': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'Cactus_1920x1080_50.yuv'),
    # 'BasketballDrive': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'BasketballDrive_1920x1080_50.yuv'),
    # 'MarketPlace': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'MarketPlace_1920x1080_60fps_10bit_420.yuv'),
    # 'RitualDance': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'RitualDance_1920x1080_60fps_10bit_420.yuv'),
    # 'ArenaOfValor': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'ArenaOfValor_1920x1080_60_8bit_420.yuv'),

    # # --- 4k (CLASS A1) ---
    # 'Campfire': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'Campfire_3840x2160_30fps_bt709_420_videoRange.yuv'),
    # 'FoodMarket4': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'FoodMarket4_3840x2160_60fps_10bit_420.yuv'),
    # 'Tango2': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'Tango2_3840x2160_60fps_10bit_420.yuv'),

    # # --- 4k (CLASS A2) ---
    # 'CatRobot': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'CatRobot_3840x2160_60fps_10bit_420_jvet.yuv'),
    # 'DaylightRoad2': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'DaylightRoad2_3840x2160_60fps_10bit_420.yuv'),
    # 'ParkRunning3': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'ParkRunning3_3840x2160_50fps_10bit_420.yuv'),
}



# ==============================
# Arquivos CSV
# ==============================
csv_input_file = os.path.join(PROJECT_FOLDER, "features.csv")
csv_output_file = os.path.join(PROJECT_FOLDER, "new-features.csv")


# =======================================================
# FEATURE 1: Média, variância, desvio padrão e soma
# =======================================================
def calculate_basic_features_cv(block_float):
    """Calcula média, variância, desvio padrão e soma total."""
    mean_val, std_val = cv2.meanStdDev(block_float)
    mean_val = float(mean_val[0][0])
    std_dev_val = float(std_val[0][0])
    var_val  = std_dev_val ** 2
    sum_val = mean_val * block_float.size
    return mean_val, var_val, std_dev_val, sum_val

# =======================================================
# FEATURE 2: vH, vV, dH, dV com OpenCV
# =======================================================
def calculate_stats_cv(block_float):
    """Calcula variância e desvio padrão horizontais e verticais."""
    # blk deve ser float32 para as subtrações
    row_means  = cv2.reduce(block_float, dim=1, rtype=cv2.REDUCE_AVG)
    row_stds   = np.sqrt(cv2.reduce((block_float - row_means)**2, 1, cv2.REDUCE_AVG))
    col_means  = cv2.reduce(block_float, dim=0, rtype=cv2.REDUCE_AVG)
    col_stds   = np.sqrt(cv2.reduce((block_float - col_means)**2, 0, cv2.REDUCE_AVG))
    row_vars = row_stds**2
    col_vars = col_stds**2
    vH = float(np.mean(row_vars))
    vV = float(np.mean(col_vars))
    dH = float(np.mean(row_stds))
    dV = float(np.mean(col_stds))
    return vH, vV, dV, dH

# =======================================================
# FEATURE 3: Gradiente Sobel
# =======================================================
def calculate_gradients_sobel_cv(block_float):
    """Calcula gradientes Sobel, magnitude, direção e razão."""
    Gh = cv2.Sobel(block_float, cv2.CV_32F, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    Gv = cv2.Sobel(block_float, cv2.CV_32F, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    mGv = float(np.mean(np.abs(Gv)))
    mGh = float(np.mean(np.abs(Gh)))
    mag = cv2.magnitude(Gv, Gh)
    direction = cv2.phase(Gh, Gv, angleInDegrees=True)
    razao_grad = mGh / (mGv + 1e-6)
    return mGv, mGh, float(np.mean(mag)), float(np.mean(direction)), float(razao_grad)

# =======================================================
# FEATURE 4: Gradiente Prewitt
# =======================================================
def calculate_gradients_prewitt_cv(block_float):
    """Calcula gradientes Prewitt, magnitude, direção e razão."""
    kernel_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_gy = np.array([[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]], dtype=np.float32)
    Gh = cv2.filter2D(block_float, -1, kernel_gx, borderType=cv2.BORDER_REPLICATE)
    Gv = cv2.filter2D(block_float, -1, kernel_gy, borderType=cv2.BORDER_REPLICATE)
    mGv = float(np.mean(np.abs(Gv)))
    mGh = float(np.mean(np.abs(Gh)))
    mag = cv2.magnitude(Gv, Gh)
    direction = cv2.phase(Gh, Gv, angleInDegrees=True)
    razao_grad = mGh / (mGv + 1e-6)
    return mGv, mGh, float(np.mean(mag)), float(np.mean(direction)), float(razao_grad)

# =======================================================
# FEATURE 5: Contraste
# =======================================================
def calculate_contrast_features_cv(block_orig):
    """Calcula min, max, e range (ponta-a-ponta) usando cv2.minMaxLoc."""
    blk_min, blk_max, _, _ = cv2.minMaxLoc(block_orig)
    blk_range = blk_max - blk_min
    return float(blk_min), float(blk_max), float(blk_range)

# =======================================================
# FEATURE 6: Nitidez (Variância Laplaciana)
# =======================================================
def calculate_laplacian_cv(block_float):
    """
    Calcula a variância do Laplaciano, um indicador de nitidez (blur).
    Valores altos = nítido; Valores baixos = borrado.
    """
    laplacian_var = float(cv2.Laplacian(block_float, cv2.CV_32F, ksize=1).var())
    return laplacian_var

# =======================================================
# FEATURE 7: Entropia de Shannon
# =======================================================
def calculate_entropy_cv(block_orig):
    """
    Calcula a Entropia de Shannon usando cv2.calcHist.
    Mede a complexidade/textura do bloco.
    """
    # Determina o range e o número de bins baseado no bit depth
    is_10bit = np.max(block_orig) > 255

    if is_10bit:
        hist_size = 1024
        hist_range = [0, 1024]
        block_calc = block_orig.astype(np.float32)
    else:
        hist_size = 256
        hist_range = [0, 256]
        block_calc = block_orig.astype(np.uint8)

    hist = cv2.calcHist([block_calc], [0], None, [hist_size], hist_range)
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    probabilities = hist[hist>0]
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)

# =======================================================
# FEATURE 8: HADAMARD
# =======================================================

def fwht_1d(a):
    """In-place 1D Fast Walsh-Hadamard Transform (length power of 2)."""
    h = 1
    n = a.shape[0]
    while h < n:
        for i in range(0, n, h*2):
            for j in range(i, i+h):
                x = a[j]
                y = a[j+h]
                a[j] = x + y
                a[j+h] = x - y
        h *= 2
    return a

def fwht_2d(mat):
    """Aplica Hadamard 2D in-place."""
    mat = mat.astype(np.float32)
    for r in range(mat.shape[0]):
        fwht_1d(mat[r, :])
    for c in range(mat.shape[1]):
        fwht_1d(mat[:, c])
    return mat

def calculate_hadamard_block_features(block):
    """Calcula features Hadamard pedidas do bloco."""
    H = fwht_2d(block.copy())

    dc = H[0,0]
    energy_total = np.sum(H**2)
    energy_ac = energy_total - dc**2
    max_coef = np.max(H)
    min_coef = np.min(H)

    # Valores nos cantos
    top_left = H[0,0]
    top_right = H[0,-1]
    bottom_left = H[-1,0]
    bottom_right = H[-1,-1]



# ==============================
# Funções auxiliares (Leitura)
# ==============================
def extract_block(Y, x, y, block_w, block_h):
    """Extrai o bloco da matriz Y conforme (x, y, w, h)."""
    if y + block_h > Y.shape[0] or x + block_w > Y.shape[1]:
        return None
    block = Y[y:y + block_h, x:x + block_w]
    if block.size == 0:
        return None
    return block


# =============================================
# Função de Processamento Principal
# =============================================

def process_csv_with_yuv(csv_input, csv_output, video_paths_mapping):
    """Processa o CSV e adiciona features (nível de bloco) dos blocos YUV."""

    print(f"Lendo: {csv_input} em modo chunk seguro para memória...")
    cols_to_check = ['frame', 'x', 'y', 'Width', 'Height', 'FrameWidth', 'FrameHeight', 'BitDepth']
    chunk_size = 90000000
    first_chunk = True
    total_lines_processed = 0

    # --- Definição das colunas de features (nível de bloco) ---
    stats_bases = [
        'blk_pixel_mean', 'blk_pixel_variance', 'blk_pixel_std_dev', 'blk_pixel_sum', # Estatísticas Básicas
        'blk_var_h', 'blk_var_v', 'blk_std_v', 'blk_std_h', # Estatísticas Direcionais
        'blk_min', 'blk_max', 'blk_range', # Contraste
        'blk_laplacian_var', # Nitidez
        'blk_entropy' # Entropia
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

    if PRINT_BLOCK_PIXELS:
        feature_columns = base_features + hadamard_bases + ['blk_pixels']
    else:
        feature_columns = base_features + hadamard_bases

    # -----------------------------------------------------------------

    for chunk in pd.read_csv(csv_input, sep=',', chunksize=chunk_size):
        print(f"\nProcessando chunk de {len(chunk)} linhas (Total processado: {total_lines_processed})...")

        # Limpeza de dados
        for c in cols_to_check:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')

        rows_with_nan = chunk[chunk[cols_to_check].isna().any(axis=1)]
        if not rows_with_nan.empty:
            for idx, row in rows_with_nan.iterrows():
                nan_cols = row[cols_to_check].isna()
                print(f"Linha {idx} removida por NaN nas colunas: {list(nan_cols[nan_cols].index)}")
                print(row)
        chunk.dropna(subset=cols_to_check, inplace=True)
        chunk[cols_to_check] = chunk[cols_to_check].astype(int)

        # Inicializa colunas
        for col in feature_columns:
            chunk[col] = np.nan

        # Agrupa por vídeo e frame
        grouped = chunk.groupby(['video', 'frame'])

        for (video_name, frame_num), group in grouped:
            video_path = video_paths_mapping.get(video_name)
            if not video_path or not os.path.exists(video_path):
                print(f"   ERRO: vídeo '{video_name}' não encontrado em {video_path}.")
                continue

            ref = group.iloc[0]

            try:
                width = ref['FrameWidth']
                height = ref['FrameHeight']
                bit_depth = ref['BitDepth']
                if bit_depth > 8:
                    pix_fmt = 'yuv420p10le'
                else:
                    pix_fmt = 'yuv420p'
                Y, U, V = yuvio.imread(video_path, width, height, pix_fmt, frame_num)
            except Exception as e:
                csv_lines = group.index.tolist()
                print(f"[Vídeo: {video_name} | Frame: {frame_num} | CSV linhas: {csv_lines}] ERRO ao ler frame com yuvio: {e}")
                continue

            # Itera sobre as linhas do CSV (blocos)
            for i, row in group.iterrows():
                blk_orig = extract_block(Y, row['x'], row['y'], row['Width'], row['Height'])
                if blk_orig is None:
                    print(f"   ERRO: bloco não extraído para linha {i} — ignorando.")
                    continue

                blk_float = blk_orig.astype(np.float32)

                if PRINT_BLOCK_PIXELS:
                    max_elems = 200
                    try:
                        if blk_float.size <= max_elems:
                            blk_str = np.array2string(blk_float, precision=2, separator=', ', max_line_width=200)
                            print(f"  Bloco (Video={video_name} Frame={frame_num} X={row['x']} Y={row['y']} size={blk_float.shape}):")
                            print(blk_str)
                        else:
                            blk_str = np.array2string(blk_float[:5, :5], precision=2, separator=', ', max_line_width=200)
                            blk_str = f"[TRUNCATED: only top-left 5x5 shown] {blk_str}"
                            print(f"  Bloco grande (Video={video_name} Frame={frame_num} X={row['x']} Y={row['y']} size={blk_float.shape}, elements={blk_float.size}) - mostrando canto superior esquerdo 5x5:")
                            print(blk_str)
                        chunk.at[i, 'blk_pixels'] = blk_str
                    except Exception as e:
                        print(f"   ERRO ao converter bloco para string na linha {i}: {e}")
                        chunk.at[i, 'blk_pixels'] = '[ERROR]'

                # --- Início dos Cálculos (Nível de Bloco) ---
                try:
                    # 1. Features basicas (Média, Variância, StdDev, Soma)
                    mean_val, var_val, std_val, sum_val = calculate_basic_features_cv(blk_float)
                    chunk.at[i, 'blk_pixel_mean'] = mean_val
                    chunk.at[i, 'blk_pixel_variance'] = var_val
                    chunk.at[i, 'blk_pixel_std_dev'] = std_val
                    chunk.at[i, 'blk_pixel_sum'] = sum_val

                    # 2. Features de Stats (vH, vV, dV, dH)
                    vH, vV, dV, dH = calculate_stats_cv(blk_float)
                    chunk.at[i, 'blk_var_h'] = vH
                    chunk.at[i, 'blk_var_v'] = vV
                    chunk.at[i, 'blk_std_v'] = dV
                    chunk.at[i, 'blk_std_h'] = dH

                    # 3. Features de Gradiente (Sobel)
                    mGv_s, mGh_s, mag_s, dir_s, razao_s = calculate_gradients_sobel_cv(blk_float)
                    chunk.at[i, 'blk_sobel_gv'] = mGv_s
                    chunk.at[i, 'blk_sobel_gh'] = mGh_s
                    chunk.at[i, 'blk_sobel_mag'] = mag_s
                    chunk.at[i, 'blk_sobel_dir'] = dir_s
                    chunk.at[i, 'blk_sobel_razao_grad'] = razao_s

                    # 4. Features de Gradiente (Prewitt)
                    mGv_p, mGh_p, mag_p, dir_p, razao_p = calculate_gradients_prewitt_cv(blk_float)
                    chunk.at[i, 'blk_prewitt_gv'] = mGv_p
                    chunk.at[i, 'blk_prewitt_gh'] = mGh_p
                    chunk.at[i, 'blk_prewitt_mag'] = mag_p
                    chunk.at[i, 'blk_prewitt_dir'] = dir_p
                    chunk.at[i, 'blk_prewitt_razao_grad'] = razao_p

                    # 5. Contraste (OpenCV)
                    b_min, b_max, b_range = calculate_contrast_features_cv(blk_orig)
                    chunk.at[i, 'blk_min'] = b_min
                    chunk.at[i, 'blk_max'] = b_max
                    chunk.at[i, 'blk_range'] = b_range

                    # 6. Nitidez (Laplaciano)
                    chunk.at[i, 'blk_laplacian_var'] = calculate_laplacian_cv(blk_float)

                    # 7. Entropia
                    chunk.at[i, 'blk_entropy'] = calculate_entropy_cv(blk_orig)

                    # --- 8. Features Hadamard ---
                    h, w = blk_orig.shape

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
                    print(f"   AVISO: Erro de OpenCV ao processar bloco na linha {i} (Shape: {blk_orig.shape}). Provavelmente bloco menor que 3x3. {e}. Preenchendo com NaN.")
                    continue
                except Exception as e:
                    print(f"   ERRO: Erro inesperado ao processar bloco na linha {i} (Shape: {blk_orig.shape}): {e}. Preenchendo com NaN.")
                    continue
                # --- Fim dos Cálculos ---

        # Salva o chunk processado
        print(f"Salvando chunk no disco...")
        chunk.to_csv(csv_output, mode='w' if first_chunk else 'a', header=first_chunk, index=False, sep=';')
        first_chunk = False
        total_lines_processed += len(chunk)

    print(f"\n Processamento concluído. Arquivo final salvo em: {csv_output}")


# ==============================
# Execução principal
# ==============================
if __name__ == "__main__":
    process_csv_with_yuv(csv_input_file, csv_output_file, video_paths_yuv)