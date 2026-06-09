import os
from PIL import Image
from tqdm import tqdm

frame_name = '600_47n8_1_2mm_-161_08_42_12_287'
# --- KONFIGURACJA ---
SEQUENCES_DIR = "frames_output"
# OUTPUT_ARC = "output_rois/train_roi_arc"
# OUTPUT_WELD = "output_rois/train_roi_weld"
# LOG_PATH = "roi_extraction_log.txt"
OUTPUT_ARC = f"output_rois/{frame_name}_test_roi_arc"
OUTPUT_WELD = f"output_rois/{frame_name}_test_roi_weld"
LOG_PATH = f"{frame_name}_test_roi_extraction_log.txt"

# Lista sekwencji do przetworzenia (foldery z frames_output)
SELECTED_SEQUENCES = [
    frame_name,
]

# Współrzędne ROI — możesz zmienić
ROI_ARC = (230, 420, 345, 480)
ROI_WELD = (230, 0, 345, 420)

# --- UTWÓRZ FOLDERY I LOG ---
os.makedirs(OUTPUT_ARC, exist_ok=True)
os.makedirs(OUTPUT_WELD, exist_ok=True)
log_file = open(LOG_PATH, "w")

# --- ZBIERZ OBRAZY TYLKO Z WYBRANYCH SEKWENCJI ---
image_paths = []
for seq_folder in SELECTED_SEQUENCES:
    seq_path = os.path.join(SEQUENCES_DIR, seq_folder, "preview_fixed")
    if not os.path.isdir(seq_path):
        print(f"⚠️  Pominięto: {seq_path} (nie istnieje)")
        continue
    for fname in os.listdir(seq_path):
        if fname.lower().endswith((".jpg", ".png")):
            image_paths.append((seq_folder, os.path.join(seq_path, fname)))

# --- PROGRESS BAR I EKSTRAKCJA ---
for seq_name, fpath in tqdm(image_paths, desc="Ekstrakcja ROIs"):
    fname = os.path.basename(fpath)
    try:
        with Image.open(fpath) as img:
            roi_arc = img.crop(ROI_ARC)
            roi_weld = img.crop(ROI_WELD)

            out_arc = os.path.join(OUTPUT_ARC, f"{seq_name}_arc_{fname}")
            out_weld = os.path.join(OUTPUT_WELD, f"{seq_name}_weld_{fname}")

            roi_arc.save(out_arc)
            roi_weld.save(out_weld)

            log_file.write(f"{fname} OK ({seq_name})\n")
    except Exception as e:
        log_file.write(f"{fname} ERROR: {e}\n")

log_file.close()
print(f"✅ Gotowe. Przetworzono {len(image_paths)} obrazów. Log: {LOG_PATH}")
