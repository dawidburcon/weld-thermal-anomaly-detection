# import cv2
# import numpy as np
# import time
# import os


# seq_file_name = "625_38n18_1_2mm_-161_07_41_19_806"  # Nazwa sekwencji, którą chcesz przetworzyć
# # === PARAMETRY ===
# THRESHOLD = 0.347504
# ROI = (230, 0, 345, 420)  # (x, y, width, height)
# DELAY = 100  # ms between frames
# SOURCE_TYPE = "video"  # "video" or "frames"
# VIDEO_PATH = f"frames_ouput/{seq_file_name}.mp4"  # ścieżka do pliku wideo
# FRAMES_DIR = f"frames_ouput/{seq_file_name}/preview_fixed"  # folder z klatkami (jpg/png)

# def binarize(image, threshold):
#     """Zamienia obraz na binarny (0/255) na podstawie RGB sumy i progu"""
#     gray = np.sum(image, axis=2)  # suma RGB
#     max_val = 255 * 3
#     binary = (gray >= max_val * threshold).astype(np.uint8) * 255
#     return binary

# def process_frame(frame, roi, threshold):
#     x, y, w, h = roi
#     cropped = frame[y:y+h, x:x+w]
#     binarized = binarize(cropped, threshold)
#     return cropped, binarized

# def run_simulation():
#     if SOURCE_TYPE == "video":
#         cap = cv2.VideoCapture(VIDEO_PATH)
#         if not cap.isOpened():
#             print("Nie można otworzyć pliku wideo.")
#             return

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             roi_orig, roi_bin = process_frame(frame, ROI, THRESHOLD)

#             # Pokaż obie wersje obok siebie
#             combined = np.hstack([roi_orig, cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)])
#             cv2.imshow("Monitoring spoiny", combined)

#             if cv2.waitKey(DELAY) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     elif SOURCE_TYPE == "frames":
#         frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

#         for fname in frames:
#             path = os.path.join(FRAMES_DIR, fname)
#             frame = cv2.imread(path)
#             if frame is None:
#                 continue

#             roi_orig, roi_bin = process_frame(frame, ROI, THRESHOLD)
#             combined = np.hstack([roi_orig, cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)])
#             cv2.imshow("Monitoring spoiny", combined)

#             if cv2.waitKey(DELAY) & 0xFF == ord('q'):
#                 break

#         cv2.destroyAllWindows()

#     else:
#         print("Nieznany typ źródła:", SOURCE_TYPE)

# if __name__ == "__main__":
#     run_simulation()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# === KONFIGURACJA ===
video_path = 'twoja_sekwencja.mp4'  # <-- Podaj ścieżkę do pliku wideo
mask_path = 'images/maska.png'

# === PROGI klasyfikacji ===
geom_thresh_contours = 10
geom_thresh_area = 200
term_thresh_percent = 2.0
term_thresh_area = 30

# === Kolory klas
color_map = {
    "OK": (0, 255, 0),
    "TERM_WADA": (0, 165, 255),
    "GEOM_WADA": (255, 0, 0),
    "MIESZANA": (0, 0, 255)
}

# === Wczytaj maskę spoiny
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise ValueError("❌ Nie można wczytać maski!")

# === Inicjalizacja wideo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("❌ Nie można otworzyć pliku wideo!")

# === Wykres klasy jakości w czasie
plt.ion()
fig, ax = plt.subplots(figsize=(10, 2))
frame_nums = deque(maxlen=100)
quality_labels = deque(maxlen=100)

def plot_quality_trend(frame_nums, quality_labels):
    ax.clear()
    color_codes = {'OK': 'green', 'TERM_WADA': 'orange', 'GEOM_WADA': 'blue', 'MIESZANA': 'red'}
    ax.scatter(frame_nums, [1]*len(frame_nums), c=[color_codes[q] for q in quality_labels], s=200, marker='|')
    ax.set_title("Trend jakości spoiny (ostatnie 100 klatek)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.pause(0.001)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Koniec wideo.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if mask.shape != gray.shape:
        resized_mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = mask.copy()
    _, mask_bin = cv2.threshold(resized_mask, 127, 1, cv2.THRESH_BINARY)

    temps = gray[mask_bin == 1]
    if len(temps) == 0:
        continue

    # === Detekcja termiczna
    mean = np.mean(temps)
    std = np.std(temps)
    anomaly_mask = np.zeros_like(mask, dtype=np.uint8)
    anomaly_pixels = (gray > mean + 2 * std) | (gray < mean - 2 * std)
    anomaly_mask[(mask_bin == 1) & anomaly_pixels] = 255

    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_temp = len(contours)
    ratio = (np.sum(anomaly_mask == 255) / np.sum(mask_bin)) * 100
    area_mean = np.sum([cv2.contourArea(c) for c in contours]) / num_temp if num_temp > 0 else 0

    # === Detekcja geometryczna (Canny)
    gray8 = gray.astype(np.uint8)
    edges = cv2.Canny(gray8, 50, 150)
    edges_in_mask = cv2.bitwise_and(edges, edges, mask=mask_bin.astype(np.uint8))
    g_contours, _ = cv2.findContours(edges_in_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_geom = len(g_contours)
    area_geom = np.sum([cv2.contourArea(c) for c in g_contours])

    # === Klasyfikacja jakości
    is_geom_bad = (num_geom > geom_thresh_contours) or (area_geom > geom_thresh_area)
    is_term_bad = (ratio > term_thresh_percent) or (area_mean > term_thresh_area)

    if is_term_bad and is_geom_bad:
        quality = "MIESZANA"
    elif is_term_bad:
        quality = "TERM_WADA"
    elif is_geom_bad:
        quality = "GEOM_WADA"
    else:
        quality = "OK"

    # === Wyświetlanie klatki z opisem
    annotated = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    cv2.putText(annotated, f"{quality}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_map[quality], 3)

    cv2.imshow("Analiza spoiny", annotated)

    # === Aktualizacja wykresu trendu
    frame_nums.append(frame_id)
    quality_labels.append(quality)
    plot_quality_trend(frame_nums, quality_labels)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    frame_id += 1

# === Koniec
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

