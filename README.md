# yuv-feature-engineering
Feature extraction and engineering from YUV images for video analysis and machine learning tasks.


# ======================================================================
# DESCRIÇÃO DAS FEATURES CALCULADAS
# ======================================================================

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