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
h# -*- coding: utf-8 -*-
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
seq_file_name = '600_56n17_1mm_-161_09_29_59_808'
# seq_file_name = '600_41n20_1_2mm_-161_08_03_50_784'
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
df = pd.read_csv(f"./frames_output/{seq_file_name}/temperature_stats.csv")

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
anomalies.to_csv(f"./frames_output/{seq_file_name}/anomalies_detected.csv", index=False)

# %%
# import cv2
# import matplotlib.pyplot as plt
# import pandas as pd
# import os

# # Załaduj dane z wykrytymi anomaliami
# anomalies = pd.read_csv("./frames_output/625_38n18_1_2mm_-161_07_41_19_806/anomalies_detected.csv")

# # Folder z zapisanymi obrazkami podglądów
# preview_fixed_dir = "./frames_output/625_38n18_1_2mm_-161_07_41_19_806/preview_fixed/"

# # Przygotuj wykres
# fig, axes = plt.subplots(nrows=len(anomalies), ncols=2, figsize=(10, len(anomalies) * 5))

# # Upewnij się, że są odpowiednie wymiary wykresu
# if len(anomalies) == 1:
#     axes = [axes]

# # Dla każdej wykrytej anomalii
# for i, row in anomalies.iterrows():
#     frame_name = row['Frame']
    
#     # Ścieżka do podglądu obrazu
#     image_path = os.path.join(preview_fixed_dir, frame_name.replace(".tiff", ".jpg"))
    
#     # Wczytaj obrazek
#     img = cv2.imread(image_path)
    
#     # Konwertuj obrazek do RGB, bo OpenCV ładuje w BGR
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Statystyki anomalii
#     stats = f"AvgTemp: {row['AvgTemp']:.2f}°C\n" \
#             f"WeldAvgTemp: {row['WeldAvgTemp']:.2f}°C\n" \
#             f"BottomAvgTemp: {row['BottomAvgTemp']:.2f}°C\n" \
#             f"BottomHotPx: {row['BottomHotPx']}\n" \
#             f"Anomaly Score: {row['AnomalyScore']}"
    
#     # Wyświetl obrazek i statystyki
#     axes[i][0].imshow(img_rgb)
#     axes[i][0].axis('off')
#     axes[i][0].set_title(f"Anomalna Klatka: {frame_name}")
    
#     axes[i][1].text(0.5, 0.5, stats, fontsize=12, ha='center', va='center')
#     axes[i][1].axis('off')

# # Dopasuj wykresy
# plt.tight_layout()
# plt.show()


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

# %% [markdown]
# PYTORCH SHIIIIT

# %%
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Wczytaj dane
df = pd.read_csv(f"./frames_output/{seq_file_name}/temperature_stats.csv")

# Wybierz kolumny numeryczne (bez 'Frame' i Min/MaxTemp jeśli nie są istotne)
features = df[['AvgTemp', 'WeldAvgTemp', 'WeldStdTemp', 'BottomAvgTemp', 'MiddleAvgTemp']]

# Normalizacja (bardzo ważne dla autoenkodera)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Zamiana na tensory
X = torch.tensor(features_scaled, dtype=torch.float32)
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=4, shuffle=True)


# %%
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder(input_dim=X.shape[1])


# %%
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    for batch in loader:
        x_batch = batch[0]
        output = model(x_batch)
        loss = criterion(output, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# %%
# Przeanalizuj cały zbiór danych po treningu
with torch.no_grad():
    recon = model(X)
    errors = torch.mean((recon - X) ** 2, dim=1)  # MSE per sample

# Dodaj do DataFrame
df['reconstruction_error'] = errors.numpy()

# Przykład: oznacz anomalie gdy błąd > próg (np. 95 percentyl)
threshold = df['reconstruction_error'].quantile(0.95)
df['anomaly'] = df['reconstruction_error'] > threshold

df[['Frame', 'reconstruction_error', 'anomaly']]




# %%
df_anomalies = df[df['anomaly']]
df_anomalies

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['Frame'], df['reconstruction_error'], label='Reconstruction error', color='blue')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')

# Zaznaczenie anomalii
anomalies = df[df['anomaly']]
plt.scatter(anomalies['Frame'], anomalies['reconstruction_error'], color='orange', label='Anomaly', zorder=5)

plt.xlabel('Frame')
plt.ylabel('Reconstruction Error')
plt.title('Błąd rekonstrukcji w czasie + anomalie')
plt.xticks(rotation=45)

# Ustawienia osi X
step = 50
plt.xticks(ticks=range(0, len(df), step), labels=df['Frame'][::step], rotation=90)


plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# %%
df['anomaly'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Liczba ramek normalnych vs. anomalnych')
plt.xticks(ticks=[0, 1], labels=['Normalne', 'Anomalie'], rotation=0)
plt.ylabel('Liczba ramek')
plt.grid(axis='y')
plt.show()

# %%
plt.plot(df['Frame'], df['WeldAvgTemp'], label='WeldAvgTemp')
plt.xticks(rotation=45)

# Ustawienia osi X
step = 100
plt.xticks(ticks=range(0, len(df), step), labels=df['Frame'][::step], rotation=90)


plt.title('Temperatura średnia spoiny w czasie')
plt.ylabel('Temperatura')
plt.xlabel('Frame')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# --- KONFIGURACJA ---
IMAGE_DIR = f'frames_output/{seq_file_name}/preview_fixed'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- PRZYGOTOWANIE DANYCH ---
transform = transforms.Compose([
    transforms.Grayscale(),  # konwersja do 1 kanału
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),  # skala 0–1
])

class ThermalDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.files[idx])

dataset = ThermalDataset(IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- AUTOENCODER ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64
            nn.Sigmoid(),  # wyjście w zakresie 0–1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = ConvAutoencoder().to(DEVICE)


# --- TRENING ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch, _ in dataloader:
        batch = batch.to(DEVICE)
        output = model(batch)
        loss = criterion(output, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")


# --- DETEKCJA ANOMALII ---
model.eval()
reconstruction_errors = []
filenames = []

with torch.no_grad():
    for img, fname in DataLoader(dataset, batch_size=1, shuffle=False):
        img = img.to(DEVICE)
        recon = model(img)
        loss = criterion(recon, img)
        reconstruction_errors.append(loss.item())
        filenames.append(fname[0])

# --- WIZUALIZACJA ---
threshold = np.percentile(reconstruction_errors, 95)
is_anomaly = [e > threshold for e in reconstruction_errors]

plt.figure(figsize=(12, 6))
plt.plot(reconstruction_errors, label='Reconstruction error')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')

anomalies = [i for i, a in enumerate(is_anomaly) if a]
plt.scatter(anomalies, [reconstruction_errors[i] for i in anomalies], color='orange', label='Anomalie')

plt.xlabel('Obraz')
plt.ylabel('Błąd rekonstrukcji')
plt.title('Anomalie wykryte na podstawie rekonstrukcji autoencodera')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- PODGLĄD NAJWYŻSZYCH BŁĘDÓW ---
print("Obrazy potencjalnie anomalne:")
for i in anomalies:
    print(f"{filenames[i]} - error = {reconstruction_errors[i]:.4f}")


# %%
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Lista szczegółów
detailed_anomalies = []

model.eval()
with torch.no_grad():
    for i, (img, fname) in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
        img = img.to(DEVICE)
        recon = model(img)
        mse_map = F.mse_loss(recon, img, reduction='none').squeeze().cpu().numpy()
        total_error = mse_map.mean()

        if total_error > threshold:
            # Klasyfikacja typu anomalii (proste heurystyki)
            max_region_error = mse_map[24:40, 24:40].mean()  # centralna część
            outside_error = (mse_map.sum() - max_region_error * 16 * 16) / (64*64 - 256)

            if max_region_error < 0.001:  # środek ciemny = brak łuku
                anomaly_type = "Brak łuku"
            elif max_region_error < outside_error * 1.2:  # środek nie jest gorętszy niż reszta
                anomaly_type = "Łuk przesunięty"
            else:
                anomaly_type = "Inna anomalia (np. artefakt/kształt)"

            detailed_anomalies.append({
                "filename": fname[0],
                "error": total_error,
                "type": anomaly_type,
                "mse_map": mse_map,
                "image": img.squeeze().cpu().numpy(),
                "reconstruction": recon.squeeze().cpu().numpy(),
            })


# %%
for item in detailed_anomalies[:5]:  # pokaż pierwsze 5
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(item['image'], cmap='gray')
    axs[0].set_title('Oryginalny obraz')
    axs[1].imshow(item['reconstruction'], cmap='gray')
    axs[1].set_title('Rekonstrukcja')
    axs[2].imshow(item['mse_map'], cmap='hot')
    axs[2].set_title(f'Mapa błędu\n({item["type"]})')
    plt.suptitle(f"{item['filename']} - {item['type']} - Error={item['error']:.4f}")
    plt.tight_layout()
    plt.show()


# %%
from collections import Counter
counts = Counter(item['type'] for item in detailed_anomalies)

plt.bar(counts.keys(), counts.values(), color='orange')
plt.title("Rozkład typów anomalii")
plt.ylabel("Liczba przypadków")
plt.xticks(rotation=15)
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# --- KONFIGURACJA ---
IMAGE_DIR = f'frames_output/{seq_file_name}/preview_fixed'
ROI = (295, 410, 345, 480)  # lewy, górny, prawy, dolny
IMAGE_SIZE = (64, 64)  # zmniejszamy region
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- PRZYGOTOWANIE DANYCH ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

class ArcRegionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        img = img.crop(ROI)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.files[idx])

dataset = ArcRegionDataset(IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- AUTOENCODER ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = ConvAutoencoder().to(DEVICE)


# --- TRENING ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch, _ in dataloader:
        batch = batch.to(DEVICE)
        output = model(batch)
        loss = criterion(output, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")


# --- DETEKCJA ANOMALII ---
model.eval()
reconstruction_errors = []
filenames = []

with torch.no_grad():
    for img, fname in DataLoader(dataset, batch_size=1, shuffle=False):
        img = img.to(DEVICE)
        recon = model(img)
        loss = criterion(recon, img)
        reconstruction_errors.append(loss.item())
        filenames.append(fname[0])

# --- ANALIZA ---
threshold = np.percentile(reconstruction_errors, 95)
is_anomaly = [e > threshold for e in reconstruction_errors]

plt.figure(figsize=(12, 6))
plt.plot(reconstruction_errors, label='Reconstruction error')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')

anomalies = [i for i, a in enumerate(is_anomaly) if a]
plt.scatter(anomalies, [reconstruction_errors[i] for i in anomalies], color='orange', label='Anomalie')

plt.xlabel('Obraz')
plt.ylabel('Błąd rekonstrukcji')
plt.title('Anomalie w obszarze łuku')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Wykryte anomalie:")
for i in anomalies:
    print(f"{filenames[i]} — error = {reconstruction_errors[i]:.4f}")


# %%
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# --- KONFIGURACJA ---
IMAGE_DIR = f'frames_output/{seq_file_name}/preview_fixed'
# IMAGE_DIR = f'frames_output/{seq_file_name}/no_ignition'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- TRANSFORMACJE ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# --- DANE Z 2 ROIs ---
class ThermalDatasetMultiROI(Dataset):
    def __init__(self, image_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.transform = transform
        self.roi_arc = (295, 410, 345, 480)   # ROI łuku
        self.roi_weld = (270, 250, 370, 400)  # ROI spoiny

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        arc_crop = img.crop(self.roi_arc)
        weld_crop = img.crop(self.roi_weld)
        if self.transform:
            arc_crop = self.transform(arc_crop)
            weld_crop = self.transform(weld_crop)
        return arc_crop, weld_crop, os.path.basename(self.files[idx])

# --- AUTOENCODER ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- PRZYGOTOWANIE ---
dataset = ThermalDatasetMultiROI(IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model_arc = ConvAutoencoder().to(DEVICE)
model_weld = ConvAutoencoder().to(DEVICE)
optimizer_arc = optim.Adam(model_arc.parameters(), lr=0.001)
optimizer_weld = optim.Adam(model_weld.parameters(), lr=0.001)
criterion = nn.MSELoss()

# --- TRENING ---
for epoch in range(EPOCHS):
    model_arc.train()
    model_weld.train()
    loss_arc_sum, loss_weld_sum = 0.0, 0.0

    for arc, weld, _ in dataloader:
        arc = arc.to(DEVICE)
        weld = weld.to(DEVICE)

        out_arc = model_arc(arc)
        loss_arc = criterion(out_arc, arc)
        optimizer_arc.zero_grad()
        loss_arc.backward()
        optimizer_arc.step()
        loss_arc_sum += loss_arc.item()

        out_weld = model_weld(weld)
        loss_weld = criterion(out_weld, weld)
        optimizer_weld.zero_grad()
        loss_weld.backward()
        optimizer_weld.step()
        loss_weld_sum += loss_weld.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Arc Loss: {loss_arc_sum:.4f} | Weld Loss: {loss_weld_sum:.4f}")

# --- DETEKCJA ANOMALII ---
model_arc.eval()
model_weld.eval()
errors_arc = []
errors_weld = []
filenames = []

with torch.no_grad():
    for arc, weld, fname in DataLoader(dataset, batch_size=1, shuffle=False):
        arc = arc.to(DEVICE)
        weld = weld.to(DEVICE)

        recon_arc = model_arc(arc)
        recon_weld = model_weld(weld)

        err_arc = criterion(recon_arc, arc).item()
        err_weld = criterion(recon_weld, weld).item()

        errors_arc.append(err_arc)
        errors_weld.append(err_weld)
        filenames.append(fname[0])

# --- ANALIZA ---
thresh_arc = np.percentile(errors_arc, 95)
thresh_weld = np.percentile(errors_weld, 95)
anomalies_arc = [i for i, e in enumerate(errors_arc) if e > thresh_arc]
anomalies_weld = [i for i, e in enumerate(errors_weld) if e > thresh_weld]

# --- WIZUALIZACJA ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(errors_arc, label='Arc Error')
plt.axhline(y=thresh_arc, color='red', linestyle='--', label='Threshold')
plt.scatter(anomalies_arc, [errors_arc[i] for i in anomalies_arc], color='orange', label='Anomalies')
plt.title('Anomalie łuku')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(errors_weld, label='Weld Error')
plt.axhline(y=thresh_weld, color='red', linestyle='--', label='Threshold')
plt.scatter(anomalies_weld, [errors_weld[i] for i in anomalies_weld], color='orange', label='Anomalies')
plt.title('Anomalie spoiny')
plt.legend()
plt.tight_layout()
plt.show()

# --- WYDRUK ---
print("Anomalie łuku:")
for i in anomalies_arc:
    print(f"{filenames[i]} - error = {errors_arc[i]:.4f}")

print("\nAnomalie spoiny:")
for i in anomalies_weld:
    print(f"{filenames[i]} - error = {errors_weld[i]:.4f}")




# %%
plt.subplot(1,2,1)
plt.imshow(original_image.squeeze().cpu(), cmap='gray')
plt.title("Oryginał")
plt.subplot(1,2,2)
plt.imshow(reconstructed_image.squeeze().cpu(), cmap='gray')
plt.title("Rekonstrukcja")
plt.show()

# %%
torch.save(model_arc.state_dict(), "autoencoder_arc_reference.pth")
torch.save(model_weld.state_dict(), "autoencoder_weld_reference.pth")

# %%
model_arc = ConvAutoencoder().to(DEVICE)
model_weld = ConvAutoencoder().to(DEVICE)

model_arc.load_state_dict(torch.load("autoencoder_arc_reference.pth"))
model_weld.load_state_dict(torch.load("autoencoder_weld_reference.pth"))

model_arc.eval()
model_weld.eval()


# %% [markdown]
# INTERFACE LOADER

# %%
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# --- ŚCIEŻKI ---
NEW_IMAGE_DIR = f'frames_output/{seq_file_name}/preview_fixed'  # zmień na swój katalog
MODEL_ARC_PATH = 'autoencoder_arc_reference.pth'
MODEL_WELD_PATH = 'autoencoder_weld_reference.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (64, 64)

# --- TRANSFORMACJE ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# --- STRUKTURA MODELU ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- ŁADOWANIE MODELÓW ---
model_arc = ConvAutoencoder().to(DEVICE)
model_arc.load_state_dict(torch.load(MODEL_ARC_PATH))
model_arc.eval()

model_weld = ConvAutoencoder().to(DEVICE)
model_weld.load_state_dict(torch.load(MODEL_WELD_PATH))
model_weld.eval()

# --- DATASET Z ROIs ---
class ThermalDatasetMultiROI(Dataset):
    def __init__(self, image_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.transform = transform
        self.roi_arc = (295, 410, 345, 480)
        self.roi_weld = (270, 250, 370, 400)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        arc_crop = img.crop(self.roi_arc)
        weld_crop = img.crop(self.roi_weld)
        if self.transform:
            arc_crop = self.transform(arc_crop)
            weld_crop = self.transform(weld_crop)
        return arc_crop, weld_crop, os.path.basename(self.files[idx])

# --- DANE I PRZETWARZANIE ---
dataset = ThermalDatasetMultiROI(NEW_IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
criterion = nn.MSELoss()

errors_arc, errors_weld, filenames = [], [], []

with torch.no_grad():
    for arc, weld, fname in dataloader:
        arc = arc.to(DEVICE)
        weld = weld.to(DEVICE)

        recon_arc = model_arc(arc)
        recon_weld = model_weld(weld)

        err_arc = criterion(recon_arc, arc).item()
        err_weld = criterion(recon_weld, weld).item()

        errors_arc.append(err_arc)
        errors_weld.append(err_weld)
        filenames.append(fname[0])

# --- PROGI I WYKRYWANIE ---
threshold_arc = np.percentile(errors_arc, 95)
threshold_weld = np.percentile(errors_weld, 95)

anomalies_arc = [i for i, e in enumerate(errors_arc) if e > threshold_arc]
anomalies_weld = [i for i, e in enumerate(errors_weld) if e > threshold_weld]

# --- WIZUALIZACJA ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(errors_arc, label="Arc Error")
plt.axhline(threshold_arc, color='r', linestyle='--', label='Threshold')
plt.scatter(anomalies_arc, [errors_arc[i] for i in anomalies_arc], color='orange', label='Anomalies')
plt.title("Anomalie łuku")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(errors_weld, label="Weld Error")
plt.axhline(threshold_weld, color='r', linestyle='--', label='Threshold')
plt.scatter(anomalies_weld, [errors_weld[i] for i in anomalies_weld], color='orange', label='Anomalies')
plt.title("Anomalie spoiny")
plt.legend()

plt.tight_layout()
plt.show()

# --- WYDRUK ---
print("\n[ANOMALIE ŁUKU]")
for i in anomalies_arc:
    print(f"{filenames[i]} - error = {errors_arc[i]:.4f}")

print("\n[ANOMALIE SPOINY]")
for i in anomalies_weld:
    print(f"{filenames[i]} - error = {errors_weld[i]:.4f}")

