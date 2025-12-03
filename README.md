# 📌 yuv-feature-engineering

Extração e engenharia de features a partir de blocos Y de vídeos YUV para análise de vídeo e aplicações de machine learning.  
O pipeline lê um CSV com posições de blocos, carrega os frames correspondentes dos vídeos YUV e calcula um conjunto extenso de atributos estatísticos, estruturais e frequenciais.

---

# 📂 Funcionalidades Principais

- Processamento eficiente de CSV em _chunks_ (suporta arquivos muito grandes).
- Extração direta dos blocos Y (luma) a partir de vídeos YUV 4:2:0.
- Cálculo de features estatísticas, estruturais, direcionais, de textura e de transformada.
- Suporte a vídeos 8-bit e 10-bit.
- Suporte a múltiplas bases de vídeos (4k, 1080p, 720p, JVET, CTC etc.).
- Escrita incremental de novo CSV com todas as features adicionadas.

---

# 🧠 **Descrição das Features Extraídas**

A seguir estão todas as features calculadas pelo pipeline, organizadas por categoria.

---

## **1. Estatísticas Básicas (stats_bases)**

| Feature              | Descrição                        |
| -------------------- | -------------------------------- |
| `blk_pixel_mean`     | Média dos pixels do bloco.       |
| `blk_pixel_variance` | Variância dos pixels do bloco.   |
| `blk_pixel_std_dev`  | Desvio padrão.                   |
| `blk_pixel_sum`      | Soma total dos valores do bloco. |

---

## **2. Estatísticas Direcionais (stats_bases)**

| Feature     | Descrição                               |
| ----------- | --------------------------------------- |
| `blk_var_h` | Variância média por linha (horizontal). |
| `blk_var_v` | Variância média por coluna (vertical).  |
| `blk_std_h` | Desvio padrão médio por linha.          |
| `blk_std_v` | Desvio padrão médio por coluna.         |

---

## **3. Contraste e Nitidez (stats_bases)**

| Feature             | Descrição                                            |
| ------------------- | ---------------------------------------------------- |
| `blk_min`           | Valor mínimo do bloco.                               |
| `blk_max`           | Valor máximo.                                        |
| `blk_range`         | Amplitude (max − min).                               |
| `blk_laplacian_var` | Variância do Laplaciano (indicador de nitidez/blur). |

---

## **4. Complexidade / Textura**

| Feature       | Descrição                                               |
| ------------- | ------------------------------------------------------- |
| `blk_entropy` | Entropia de Shannon do bloco (complexidade da textura). |

---

## **5. Gradientes Sobel (grad_bases_sobel)**

| Feature                | Descrição                                |
| ---------------------- | ---------------------------------------- |
| `blk_sobel_gv`         | Gradiente vertical (bordas horizontais). |
| `blk_sobel_gh`         | Gradiente horizontal (bordas verticais). |
| `blk_sobel_mag`        | Magnitude média do gradiente.            |
| `blk_sobel_dir`        | Direção média (em graus).                |
| `blk_sobel_razao_grad` | Razão gh / gv.                           |

---

## **6. Gradientes Prewitt (grad_bases_prewitt)**

| Feature                  | Descrição                         |
| ------------------------ | --------------------------------- |
| `blk_prewitt_gv`         | Gradiente vertical via Prewitt.   |
| `blk_prewitt_gh`         | Gradiente horizontal via Prewitt. |
| `blk_prewitt_mag`        | Magnitude média via Prewitt.      |
| `blk_prewitt_dir`        | Direção média via Prewitt.        |
| `blk_prewitt_razao_grad` | Razão gh / gv (Prewitt).          |

---

## **7. Transformada de Hadamard (hadamard_bases)**

| Feature                | Descrição                                    |
| ---------------------- | -------------------------------------------- |
| `blk_had_dc`           | Coeficiente DC (brilho geral).               |
| `blk_had_energy_total` | Soma dos quadrados de todos os coeficientes. |
| `blk_had_energy_ac`    | Energia AC (total − DC²).                    |
| `blk_had_max`          | Maior coeficiente absoluto.                  |
| `blk_had_min`          | Menor coeficiente absoluto.                  |
| `blk_had_topleft`      | Coeficiente H[0,0] (DC).                     |
| `blk_had_topright`     | Coef. canto superior direito.                |
| `blk_had_bottomleft`   | Coef. canto inferior esquerdo.               |
| `blk_had_bottomright`  | Coef. canto inferior direito.                |

---

# 📦 **Estrutura do Projeto**

```python
project/
├── config.py # Caminhos, separadores CSV, configurações globais
├── features.py # Implementação das funções de extração de features
├── process_yuv.py # Função principal de leitura do CSV + YUV
├── README.md
└── data/
```

---

# ⚙️ **Exemplo de Configuração (config.py)**

```python
# Diretório do projeto
PROJECT_FOLDER = '/home/carolinesc/mestrado'

# Arquivos CSV de entrada e saída
csv_input_file = os.path.join(PROJECT_FOLDER, "features.csv")
csv_output_file = os.path.join(PROJECT_FOLDER, "new-features.csv")

CSV_READ_SEP = ','
CSV_WRITE_SEP = ';'
CHUNK_SIZE = 90_000_000

# Colunas obrigatórias no CSV
COL_FRAME = 'frame'
COL_X = 'x'
COL_Y = 'y'
COL_WIDTH = 'Width'
COL_HEIGHT = 'Height'
COL_FRAMEWIDTH = 'FrameWidth'
COL_FRAMEHEIGHT = 'FrameHeight'
COL_BITDEPTH = 'BitDepth'


▶️ Execução
python process_yuv.py
O novo CSV será gerado em: new-features.csv
```

---

## 🛠️ Implementação em C++

Além da versão em Python, existe a opção das mesmas rotinas de extração de features em C++ — ideal para integração direta no código do codificador 

O que a implementação C++ fornece
- Funções equivalentes às de `features.py`: média, variância, gradientes (Sobel/Prewitt), entropia, variância do Laplaciano e transformada de Hadamard.

