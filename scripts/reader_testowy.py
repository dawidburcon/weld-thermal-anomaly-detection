import imageio.v2 as imageio
import numpy as np
import cv2
import os
from glob import iglob

def load_thermal_image(frame_path):
    """
    Ładuje obraz TIFF i konwertuje na wartości typu float32.
    """
    print(f"📂 Wczytuję: {frame_path}")
    thermal_image = imageio.imread(frame_path).astype(np.float32)
    return thermal_image

def convert_to_celsius(thermal_image):
    """
    Konwertuje obraz na temperaturę w °C.
    """
    print("🔄 Konwersja na °C")
    return (thermal_image / 10) - 273.15

def normalize_temperature(thermal_image_celsius):
    """
    Normalizuje obraz do zakresu temperatur (1-99 percentyl).
    """
    print("⚖️ Normalizacja obrazu")
    min_temp, max_temp = np.percentile(thermal_image_celsius, 1), np.percentile(thermal_image_celsius, 99)
    print(f"   🔥 Zakres: {min_temp:.2f}°C - {max_temp:.2f}°C")
    
    # Normalizacja obrazu w zakresie 0-255
    normalized = np.clip(thermal_image_celsius, min_temp, max_temp)  # Odcinamy wartości poza zakresem
    normalized = (normalized - min_temp) / (max_temp - min_temp) * 255.0
    return np.uint8(normalized)

def apply_colormap(normalized_image):
    """
    Nakłada mapę kolorów na znormalizowany obraz.
    """
    print("🎨 Nakładanie mapy kolorów")
    return cv2.applyColorMap(normalized_image, cv2.COLORMAP_INFERNO)

def highlight_max_temp_point(image, thermal_image_celsius):
    """
    Zaznacza punkt o najwyższej temperaturze na obrazie.
    """
    # Znajdź współrzędne punktu o najwyższej temperaturze
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thermal_image_celsius)
    
    # Zaznacz punkt na obrazie (np. czerwony punkt o promieniu 10)
    cv2.circle(image, max_loc, 10, (0, 0, 255), 2)  # (0, 0, 255) to kolor czerwony w BGR
    return image

def save_image(colored_image, output_path):
    """
    Zapisuje obraz do pliku JPG.
    """
    print(f"✅ Zapisano: {output_path}")
    cv2.imwrite(output_path, colored_image)

def process_thermal_frames(tiff_frames, preview_fixed_dir):
    """
    Przetwarza wszystkie klatki TIFF, konwertuje je na temperatury, normalizuje, stosuje mapę kolorów,
    zaznacza punkt o najwyższej temperaturze i zapisuje wyniki do katalogu.
    """
    video_frames = []  # Lista klatek do stworzenia filmu
    
    for frame in tiff_frames:
        # Wczytanie obrazu termograficznego
        thermal_image = load_thermal_image(frame)
        
        # Konwersja do temperatury w °C
        thermal_image_celsius = convert_to_celsius(thermal_image)
        
        # Normalizacja obrazu
        normalized_image = normalize_temperature(thermal_image_celsius)
        
        # Nałożenie mapy kolorów
        colored_image = apply_colormap(normalized_image)
        
        # Zaznaczenie punktu o najwyższej temperaturze
        highlighted_image = highlight_max_temp_point(colored_image, thermal_image_celsius)
        
        # Zapisanie obrazu
        output_path = os.path.join(preview_fixed_dir, os.path.basename(frame).replace(".tiff", ".jpg"))
        save_image(highlighted_image, output_path)
        
        # Dodanie klatki do filmu
        video_frames.append(highlighted_image)
    
    # Utworzenie filmu
    video_output_path = os.path.join(preview_fixed_dir, "./frames_output/625_38n18_1_2mm_-161_07_41_19_806/video/spawanie_with_max_temp_point.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Format MP4
    fps = 30  # Można dostosować do liczby klatek na sekundę
    height, width, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    for frame in video_frames:
        video_writer.write(frame)
    
    video_writer.release()
    print(f"🎥 Utworzono film: {video_output_path}")

def main():
    # Pobranie listy TIFF
    tiff_frames = sorted(list(iglob("./frames_output/625_38n18_1_2mm_-161_07_41_19_806/radiometric/*.tiff")))

    # Folder na poprawione podglądy
    preview_fixed_dir = "./frames_output/625_38n18_1_2mm_-161_07_41_19_806/preview_fixed/"
    os.makedirs(preview_fixed_dir, exist_ok=True)

    # Przetwarzanie wszystkich klatek
    process_thermal_frames(tiff_frames, preview_fixed_dir)

if __name__ == "__main__":
    main()
