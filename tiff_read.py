import imageio.v2 as imageio
import numpy as np
import cv2
import os
from glob import iglob

# Pobranie listy TIFF
tiff_frames = sorted(list(iglob("./frames_output/625_38n18_1_2mm_-161_07_41_19_806/radiometric/*.tiff")))

# Folder na poprawione podglądy
preview_fixed_dir = "./frames_output/625_38n18_1_2mm_-161_07_41_19_806/preview_fixed/"
os.makedirs(preview_fixed_dir, exist_ok=True)

for frame in tiff_frames:
    print(f"\n📂 Przetwarzam: {frame}")

    # Wczytaj TIFF
    thermal_image = imageio.imread(frame).astype(np.float32)

    # Konwersja do temperatury w °C
    thermal_image_celsius = (thermal_image /10)  - 273.15

    # Sprawdzenie rzeczywistego zakresu temperatur
    min_temp, max_temp = np.percentile(thermal_image_celsius, 1), np.percentile(thermal_image_celsius, 99)
    print(f"   🔥 Zakres: {min_temp:.2f}°C - {max_temp:.2f}°C")

    # Normalizacja do rzeczywistego zakresu (1-99 percentyl, aby pominąć szum)
    normalized = np.clip(thermal_image_celsius, min_temp, max_temp)  # Odcinamy wartości poza zakresem
    normalized = (normalized - min_temp) / (max_temp - min_temp) * 255.0
    normalized = np.uint8(normalized)

    # Dodanie mapy kolorów
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

    # Zapis obrazu
    output_path = os.path.join(preview_fixed_dir, os.path.basename(frame).replace(".tiff", ".jpg"))
    cv2.imwrite(output_path, colored)

    print(f"   ✅ Zapisano: {output_path}")

