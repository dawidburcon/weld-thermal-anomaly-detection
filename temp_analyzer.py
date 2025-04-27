# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: thermal_3.10
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Ścieżka do pliku
seq_file_name = '625_38n18_1_2mm_-161_07_41_19_806'
csv_path = f'./frames_output/{seq_file_name}/temperature_stats.csv'

# %%
# Wczytaj dane
df = pd.read_csv(csv_path)

# %%
df.head()

# %%
plt.figure(figsize=(14,6))
plt.plot(df['Frame'], df['AvgTemp'], label='Średnia temperatura')
plt.plot(df['Frame'], df['WeldAvgTemp'], label='Temperatura spawu')
plt.plot(df['Frame'], df['BottomAvgTemp'], label='Temperatura dolna')
plt.xticks(rotation=90)
plt.xlabel('Klatka')
plt.ylabel('Temperatura [°C]')
plt.title('Zmiany temperatury podczas spawania')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %%
# df['FrameNumber'] = df['Frame'].apply(lambda x: int(x.split('_')[1].split('.')[0]))

# # I teraz zamiast 'Frame' użyj 'FrameNumber' jako X
# plt.figure(figsize=(20,10))
# plt.plot(df['FrameNumber'], df['AvgTemp'], label='Średnia temperatura')

# %%
# Oblicz różnicę temperatury między kolejnymi klatkami
df['TempDiff'] = df['AvgTemp'].diff()

# Wykryj anomalie: różnica większa niż próg
threshold = 10  # możesz dostroić
anomalies = df[df['TempDiff'].abs() > threshold]

print("Wykryte anomalie:")
print(anomalies)


# %%
plt.figure(figsize=(20,10))
plt.plot(df['Frame'], df['AvgTemp'], label='Średnia temperatura')
plt.scatter(anomalies['Frame'], anomalies['AvgTemp'], color='red', label='Anomalie')

# Ustawienia osi X
step = 20
plt.xticks(ticks=range(0, len(df), step), labels=df['Frame'][::step], rotation=90)

plt.xticks(rotation=90)
plt.xlabel('Klatka')
plt.ylabel('Temperatura [°C]')
plt.title('Detekcja anomalii temperatury')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %%
from PIL import Image
import os

image_folder = f'./frames_output/{seq_file_name}/preview_fixed/'

for frame_name in anomalies['Frame']:
    # Zamień nazwę pliku .tiff na .jpg (jeśli masz JPG-i)
    frame_name_jpg = frame_name.replace('.tiff', '.jpg')
    
    # Pełna ścieżka do pliku
    image_path = os.path.join(image_folder, frame_name_jpg)
    
    # Sprawdź czy plik istnieje
    if os.path.exists(image_path):
        # Wczytaj obraz
        img = Image.open(image_path)
        
        # Wyświetl obraz
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.title(f'Anomalia: {frame_name_jpg}')
        plt.axis('off')
        plt.show()
    else:
        print(f"Nie znaleziono pliku: {frame_name_jpg}")

# %%
# Załaduj dane
df = pd.read_csv("./frames_output/625_38n18_1_2mm_-161_07_41_19_806/temperature_stats.csv")

# Policz zmiany pomiędzy klatkami
df['AvgTempDiff'] = df['AvgTemp'].diff()
df['WeldAvgTempDiff'] = df['WeldAvgTemp'].diff()
df['BottomHotPxDiff'] = df['BottomHotPx'].diff()

# --- Funkcja scoringu anomalii ---
def calculate_anomaly_score(row, thresholds):
    score = 0
    
    # Sprawdzanie poszczególnych warunków
    if abs(row['AvgTempDiff']) > thresholds['AvgTempDiff']:
        score += 1
    if abs(row['WeldAvgTempDiff']) > thresholds['WeldAvgTempDiff']:
        score += 1
    if abs(row['BottomHotPxDiff']) > thresholds['BottomHotPxDiff']:
        score += 1
    if row['WeldStdTemp'] > thresholds['WeldStdTemp']:
        score += 1
    if row['BottomAvgTemp'] < thresholds['BottomAvgTempLow']:
        score += 1
        
    return score

# --- Definicja progów ---
thresholds = {
    'AvgTempDiff': 10,          # skok średniej temperatury >10°C
    'WeldAvgTempDiff': 8,       # zmiana temperatury spoiny >8°C
    'BottomHotPxDiff': 5,       # zmiana liczby gorących pikseli >5
    'WeldStdTemp': 80,          # odchylenie temperatury spoiny >80°C
    'BottomAvgTempLow': 300     # średnia dolna temperatura <300°C (podejrzenie gaśnięcia łuku)
}

# --- Zastosowanie scoringu ---
df['AnomalyScore'] = df.apply(lambda row: calculate_anomaly_score(row, thresholds), axis=1)

# --- Oznacz anomalie ---
df['Anomaly'] = df['AnomalyScore'] >= 2  # uznajemy za anomalię jeśli score >= 2

# --- Wypisz anomalne klatki ---
anomalies = df[df['Anomaly']]

print(f"\n🔎 Wykryto {len(anomalies)} anomalnych klatek.")
print(anomalies[['Frame', 'AvgTemp', 'WeldAvgTemp', 'BottomAvgTemp', 'BottomHotPx', 'AnomalyScore']])

# --- (opcjonalnie) Zapisz anomalie do osobnego CSV ---
anomalies.to_csv("./frames_output/625_38n18_1_2mm_-161_07_41_19_806/anomalies_detected.csv", index=False)

# %%
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os

# Załaduj dane z wykrytymi anomaliami
anomalies = pd.read_csv("./frames_output/625_38n18_1_2mm_-161_07_41_19_806/anomalies_detected.csv")

# Folder z zapisanymi obrazkami podglądów
preview_fixed_dir = "./frames_output/625_38n18_1_2mm_-161_07_41_19_806/preview_fixed/"

# Przygotuj wykres
fig, axes = plt.subplots(nrows=len(anomalies), ncols=2, figsize=(10, len(anomalies) * 5))

# Upewnij się, że są odpowiednie wymiary wykresu
if len(anomalies) == 1:
    axes = [axes]

# Dla każdej wykrytej anomalii
for i, row in anomalies.iterrows():
    frame_name = row['Frame']
    
    # Ścieżka do podglądu obrazu
    image_path = os.path.join(preview_fixed_dir, frame_name.replace(".tiff", ".jpg"))
    
    # Wczytaj obrazek
    img = cv2.imread(image_path)
    
    # Konwertuj obrazek do RGB, bo OpenCV ładuje w BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Statystyki anomalii
    stats = f"AvgTemp: {row['AvgTemp']:.2f}°C\n" \
            f"WeldAvgTemp: {row['WeldAvgTemp']:.2f}°C\n" \
            f"BottomAvgTemp: {row['BottomAvgTemp']:.2f}°C\n" \
            f"BottomHotPx: {row['BottomHotPx']}\n" \
            f"Anomaly Score: {row['AnomalyScore']}"
    
    # Wyświetl obrazek i statystyki
    axes[i][0].imshow(img_rgb)
    axes[i][0].axis('off')
    axes[i][0].set_title(f"Anomalna Klatka: {frame_name}")
    
    axes[i][1].text(0.5, 0.5, stats, fontsize=12, ha='center', va='center')
    axes[i][1].axis('off')

# Dopasuj wykresy
plt.tight_layout()
plt.show()


# %%
# Konwersja kolumny Frame do sortowania
df['FrameID'] = df['Frame'].str.extract(r'(\d+)').astype(int)
df = df.sort_values(by='FrameID')

# %%
# Styl wykresów
sns.set(style="whitegrid")
plt.figure(figsize=(15, 8))

# %%
# Wykres: Średnia temperatura spoiny
plt.subplot(2, 2, 1)
sns.lineplot(data=df, x='FrameID', y='WeldAvgTemp')
plt.title("Średnia temperatura spoiny")
plt.xlabel("Klatka")
plt.ylabel("°C")

# %%
# Wykres: Odchylenie standardowe (zmienność)
plt.subplot(2, 2, 2)
sns.lineplot(data=df, x='FrameID', y='WeldStdTemp')
plt.title("Odchylenie standardowe temperatury spoiny")
plt.xlabel("Klatka")
plt.ylabel("°C")

# %%
# Wykres: Temperatura łuku
plt.subplot(2, 2, 3)
sns.lineplot(data=df, x='FrameID', y='BottomAvgTemp')
plt.title("Średnia temperatura łuku (dolna część)")
plt.xlabel("Klatka")
plt.ylabel("°C")

# %%
# Wykres: Ilość gorących pikseli
plt.subplot(2, 2, 4)
sns.lineplot(data=df, x='FrameID', y='BottomHotPx')
plt.title("Liczba pikseli >500°C w dolnej części")
plt.xlabel("Klatka")
plt.ylabel("Liczba pikseli")

# %%
plt.tight_layout()
plt.show()
