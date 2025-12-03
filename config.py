import os

# Flag to enable/disable printing of block pixels
PRINT_BLOCK_PIXELS = False

# Directory configuration
PROJECT_FOLDER = '/home/carolinesc/mestrado' # '/home/yasminsc/mestrado'

# CSV files
csv_input_file = os.path.join(PROJECT_FOLDER, "features.csv") # compiled_dataset_medium.csv
csv_output_file = os.path.join(PROJECT_FOLDER, "new-features.csv") # features_from_image.csv

# CSV / processing configuration
CSV_READ_SEP = ',' # ;
CSV_WRITE_SEP = ';'
CHUNK_SIZE = 90000000

# Required CSV column names
COL_VIDEO = 'video' # VideoName
COL_FRAME = 'frame' # Frame
COL_X = 'x' # X_Pos
COL_Y = 'y' # Y_Pos
COL_WIDTH = 'Width' # BlockWidth
COL_HEIGHT = 'Height' # BlockHeight
COL_FRAMEWIDTH = 'FrameWidth'
COL_FRAMEHEIGHT = 'FrameHeight'
COL_BITDEPTH = 'BitDepth'

COLS_TO_CHECK = [COL_FRAME, COL_X, COL_Y, COL_WIDTH, COL_HEIGHT, COL_FRAMEWIDTH, COL_FRAMEHEIGHT, COL_BITDEPTH]

# Video directories
YUV_BASE_FOLDER = '/data/videos/'
YUV_VIDEO_FOLDERS = {
    '4k': os.path.join(YUV_BASE_FOLDER, '4k'),
    '4k_jvet': os.path.join(YUV_BASE_FOLDER, '4k_jvet'),
    '1080p': os.path.join(YUV_BASE_FOLDER, '1080p'),
    '720p': os.path.join(YUV_BASE_FOLDER, '720p'),
}


# ==============================
# Mapping of YUV videos (examples)
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
    # --- 240p (CLASS D) ---
    'BQSquare': os.path.join(YUV_VIDEO_FOLDERS['240p'], 'BQSquare_416x240_60.yuv'),
    'BlowingBubbles': os.path.join(YUV_VIDEO_FOLDERS['240p'], 'BlowingBubbles_416x240_50.yuv'),
    'BasketballPass': os.path.join(YUV_VIDEO_FOLDERS['240p'], 'BasketballPass_416x240_50.yuv'),
    'RaceHorses': os.path.join(YUV_VIDEO_FOLDERS['240p'], 'RaceHorses_416x240_30.yuv'),

    # --- 480p (CLASS C) ---
    'BQMall': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'BQMall_832x480_60.yuv'),
    'BasketballDrill': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'BasketballDrill_832x480_50.yuv'),
    'RaceHorsesC': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'RaceHorsesC_832x480_30.yuv'),
    'PartyScene': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'PartyScene_832x480_50.yuv'),
    'BasketballDrillText': os.path.join(YUV_VIDEO_FOLDERS['480p'], 'BasketballDrillText_832x480_50.yuv'),

    # --- 720p (CLASS E) ---
    'FourPeople': os.path.join(YUV_BASE_FOLDER, 'FourPeople_1280x720_60.yuv'),
    'Johnny': os.path.join(YUV_BASE_FOLDER, 'Johnny_1280x720_60.yuv'),
    'KristenAndSara': os.path.join(YUV_BASE_FOLDER, 'KristenAndSara_1280x720_60.yuv'),
    'SlideEditing': os.path.join(YUV_BASE_FOLDER, 'SlideEditing_1280x720_30.yuv'),
    'SlideShow': os.path.join(YUV_BASE_FOLDER, 'SlideShow_1280x720_20.yuv'),

    # --- 1080p (CLASS B) ---
    'BQTerrace': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'BQTerrace_1920x1080_60.yuv'),
    'Cactus': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'Cactus_1920x1080_50.yuv'),
    'BasketballDrive': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'BasketballDrive_1920x1080_50.yuv'),
    'MarketPlace': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'MarketPlace_1920x1080_60fps_10bit_420.yuv'),
    'RitualDance': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'RitualDance_1920x1080_60fps_10bit_420.yuv'),
    'ArenaOfValor': os.path.join(YUV_VIDEO_FOLDERS['1080p'], 'ArenaOfValor_1920x1080_60_8bit_420.yuv'),

    # --- 4k (CLASS A1) ---
    'Campfire': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'Campfire_3840x2160_30fps_bt709_420_videoRange.yuv'),
    'FoodMarket4': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'FoodMarket4_3840x2160_60fps_10bit_420.yuv'),
    'Tango2': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'Tango2_3840x2160_60fps_10bit_420.yuv'),

    # --- 4k (CLASS A2) ---
    'CatRobot': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'CatRobot_3840x2160_60fps_10bit_420_jvet.yuv'),
    'DaylightRoad2': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'DaylightRoad2_3840x2160_60fps_10bit_420.yuv'),
    'ParkRunning3': os.path.join(YUV_VIDEO_FOLDERS['4k'], 'ParkRunning3_3840x2160_50fps_10bit_420.yuv'),
}