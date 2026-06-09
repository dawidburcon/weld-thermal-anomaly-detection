import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Wczytanie danych z CSV
seq_file_name = '625_38n18_1_2mm_-161_07_41_19_806'
csv_path = f'./frames_output/{seq_file_name}/temperature_stats.csv'
data = pd.read_csv(csv_path)

# Usuwamy kolumnę 'Frame' (bo to jest identyfikator obrazu)
data = data.drop(columns=['Frame'])

# Normalizacja danych
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Konwersja danych do tensora PyTorch
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

# Przygotowanie zbioru danych
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definicja Autoenkodera
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(8, 4),  # Wejście ma 8 cech (kolumn)
            nn.ReLU(),
            nn.Linear(4, 2)  # Kompresja do 2 wymiarów
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8)  # Rekonstrukcja do 8 cech
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Inicjalizacja modelu
model = Autoencoder()

# Funkcja straty i optymalizator
criterion = nn.MSELoss()  # Błąd średniokwadratowy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie modelu
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for data_batch in dataloader:
        inputs = data_batch[0]

        # Przewidywanie
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Błąd rekonstrukcji

        # Optymalizacja
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Po treningu, sprawdzamy jak model radzi sobie z danymi testowymi
model.eval()
with torch.no_grad():
    test_data = data_tensor  # Możemy używać tych samych danych do wykrywania anomalii
    reconstructed = model(test_data)

    # Obliczanie błędu rekonstrukcji
    reconstruction_error = torch.mean((reconstructed - test_data) ** 2, dim=1)

    # Wyświetlenie błędów rekonstrukcji
    print("Błąd rekonstrukcji:", reconstruction_error)

# Wykrywanie anomalii
threshold = 0.1  # Można dostosować próg w zależności od błędów
anomalies = reconstruction_error > threshold
print("Anomalie:", anomalies)
