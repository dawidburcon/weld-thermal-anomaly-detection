import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżka do pliku
seq_file_name = '625_38n18_1_2mm_-161_07_41_19_806'
csv_path = f'./frames_output/{seq_file_name}/temperature_stats.csv'

# Wczytaj dane
df = pd.read_csv(csv_path)

# Konwersja kolumny Frame do sortowania
df['FrameID'] = df['Frame'].str.extract(r'(\d+)').astype(int)
df = df.sort_values(by='FrameID')

# Styl wykresów
sns.set(style="whitegrid")
plt.figure(figsize=(15, 8))

# Wykres: Średnia temperatura spoiny
plt.subplot(2, 2, 1)
sns.lineplot(data=df, x='FrameID', y='WeldAvgTemp')
plt.title("Średnia temperatura spoiny")
plt.xlabel("Klatka")
plt.ylabel("°C")

# Wykres: Odchylenie standardowe (zmienność)
plt.subplot(2, 2, 2)
sns.lineplot(data=df, x='FrameID', y='WeldStdTemp')
plt.title("Odchylenie standardowe temperatury spoiny")
plt.xlabel("Klatka")
plt.ylabel("°C")

# Wykres: Temperatura łuku
plt.subplot(2, 2, 3)
sns.lineplot(data=df, x='FrameID', y='BottomAvgTemp')
plt.title("Średnia temperatura łuku (dolna część)")
plt.xlabel("Klatka")
plt.ylabel("°C")

# Wykres: Ilość gorących pikseli
plt.subplot(2, 2, 4)
sns.lineplot(data=df, x='FrameID', y='BottomHotPx')
plt.title("Liczba pikseli >500°C w dolnej części")
plt.xlabel("Klatka")
plt.ylabel("Liczba pikseli")

plt.tight_layout()
plt.show()
