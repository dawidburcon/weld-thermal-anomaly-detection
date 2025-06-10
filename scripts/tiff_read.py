import imageio.v2 as imageio
import numpy as np
import cv2
import os
from glob import iglob
import csv

# seq_file_name = '625_38n18_1_2mm_-161_07_41_19_806'
seq_file_name = '600_41n20_1_2mm_-161_08_03_50_784'

# Pobranie listy TIFF
tiff_frames = sorted(list(iglob(f"./frames_output/{seq_file_name}/radiometric/*.tiff")))

# Folder na poprawione podglądy
preview_fixed_dir = f"./frames_output/{seq_file_name}/preview_fixed/"
os.makedirs(preview_fixed_dir, exist_ok=True)

# Ścieżka do pliku CSV
csv_output_path = f"./frames_output/{seq_file_name}/temperature_stats.csv"

# Przygotuj plik CSV i zapisz nagłówki
with open(csv_output_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Frame", "MinTemp", "MaxTemp", "AvgTemp",
        "WeldAvgTemp", "WeldStdTemp",
        "BottomAvgTemp", "BottomHotPx",
        "MiddleAvgTemp"
    ])

for frame in tiff_frames:
    print(f"\n📂 Przetwarzam: {frame}")

    # Wczytaj TIFF
    thermal_image = imageio.imread(frame).astype(np.float32)

    # Konwersja do temperatury w °C
    thermal_image_celsius = (thermal_image / 10) - 273.15

    h, w = thermal_image_celsius.shape

    # --- Strefy analizy ---
    weld_band = thermal_image_celsius[:, w//2 - 40 : w//2 + 40]  # ~środkowy pas (spoina)
    bottom_band = weld_band[h - 40 :, :]                         # dolna część (łuk)
    middle_band = weld_band[200:400, :]                          # środkowy przetop

    # --- Oblicz metryki ---
    min_temp_full = np.min(thermal_image_celsius)
    max_temp_full = np.max(thermal_image_celsius)
    avg_temp_full = np.mean(thermal_image_celsius)

    weld_avg_temp = np.mean(weld_band)
    weld_std_temp = np.std(weld_band)

    bottom_avg_temp = np.mean(bottom_band)
    bottom_hot_px = np.sum(bottom_band > 500)  # Próg np. 500°C – do modyfikacji

    middle_avg_temp = np.mean(middle_band)

    # --- Normalizacja do podglądu ---
    min_temp, max_temp = np.percentile(thermal_image_celsius, 1), np.percentile(thermal_image_celsius, 99)
    normalized = np.clip(thermal_image_celsius, min_temp, max_temp)
    normalized = ((normalized - min_temp) / (max_temp - min_temp)) * 255.0
    normalized = np.uint8(normalized)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

    # --- Zapis podglądu ---
    output_path = os.path.join(preview_fixed_dir, os.path.basename(frame).replace(".tiff", ".jpg"))
    cv2.imwrite(output_path, colored)
    print(f"   ✅ Zapisano: {output_path}")

    # --- Zapis do CSV ---
    with open(csv_output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            os.path.basename(frame), min_temp_full, max_temp_full, avg_temp_full,
            weld_avg_temp, weld_std_temp,
            bottom_avg_temp, bottom_hot_px,
            middle_avg_temp
        ])

    print(f"   📊 Statystyki zapisane do: {csv_output_path}")
