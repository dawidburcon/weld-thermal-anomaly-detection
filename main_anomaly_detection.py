import os
import cv2
import torch
import numpy as np
from datetime import datetime
from torchvision import transforms
from weld_model import ConvAutoencoder           # architektura AE
from PIL import Image
import logging

# ── KONFIGURACJA ─────────────────────────────────────────────────────────
MODEL_PATH     = "models/weld_autoencoder_test.pth"
THRESHOLD_PATH = "models/weld_threshold.txt"
IMAGE_DIR      = "frames_output/625_38n18_1_2mm_-161_07_41_19_806/preview_fixed"
LOG_DIR        = "logs/anomalies"
ROI_WELD       = (230, 0, 345, 420)            # (x1, y1, x2, y2)
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── LOGGER ───────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "system.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnomalyDetector")

# ── WCZYTYWANIE MODELU I PROGU ───────────────────────────────────────────
with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read().strip())

model = ConvAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def max_difference(a, b):
    """Największa wartość bezwzględnego błędu pikseli."""
    return torch.abs(a - b).max().item()

# ── FOLDERY WYNIKOWE ─────────────────────────────────────────────────────
for sub in ("preview", "processed", "csv"):
    os.makedirs(os.path.join(LOG_DIR, sub), exist_ok=True)

# ── ANALIZA WSZYSTKICH KLATEK ────────────────────────────────────────────
frame_files = sorted(
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))
)

for idx, fname in enumerate(frame_files):
    frame_path = os.path.join(IMAGE_DIR, fname)
    image = cv2.imread(frame_path)
    if image is None:
        logger.warning(f"Nie udało się wczytać pliku: {fname}")
        continue

    # 1) ROI spoiny → tensor
    roi = image[ROI_WELD[1]:ROI_WELD[3], ROI_WELD[0]:ROI_WELD[2]]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_tensor = transform(Image.fromarray(roi_rgb)).unsqueeze(0).to(DEVICE)

    # 2) Rekonstrukcja
    with torch.no_grad():
        recon = model(roi_tensor)
    diff_val = max_difference(roi_tensor, recon)

    # 3) Decyzja
    is_anomaly = diff_val > THRESHOLD
    status_str = "ANOMALY" if is_anomaly else "OK"

    # 4) Log (zawsze)
    logger.info(
        f"frame={fname} | reconstruction_diff={diff_val:.6f} | anomaly={is_anomaly}"
    )

    # 5) Jeśli anomalia – dodatkowe zapisy
    if is_anomaly:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{os.path.splitext(fname)[0]}_{ts}"

        # oryginał
        cv2.imwrite(os.path.join(LOG_DIR, "preview", f"{base}.jpg"), image)

        # === KONWERSJA DO SKALI SZAROŚCI TYLKO RAZ ===
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # === WYCIĘCIE ROI Z SZAROŚCI ===
        roi_gray = gray_full[ROI_WELD[1]:ROI_WELD[3], ROI_WELD[0]:ROI_WELD[2]]

        # === CANNY NA ROI ===
        roi_blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        roi_edges = cv2.Canny(roi_blurred, 50, 120)

        # === ZAPIS WYNIKU ===
        cv2.imwrite(os.path.join(LOG_DIR, "processed", f"{base}_edges.jpg"), roi_edges)
        np.savetxt(os.path.join(LOG_DIR, "csv", f"{base}.csv"), roi_gray, fmt='%d', delimiter=',')


logger.info("Zakończono analizę wszystkich klatek.")
