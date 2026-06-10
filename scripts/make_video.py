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
import cv2
import os
from glob import glob

# 🔧 Podaj nazwę folderu .seq (bez ścieżki)
nazwa_seq = '625_38n18_1_2mm_-161_07_41_19_806'  # <-- <- TUTAJ PODAJ WŁASNY FOLDER

# 🔍 Ścieżka do folderu z obrazami
folder_path = os.path.join('./frames_output', nazwa_seq, 'preview_fixed')

# 📦 Pobierz i posortuj pliki .jpg
images = sorted(glob(os.path.join(folder_path, '*.jpg')))
if not images:
    print(f'Brak plików .jpg w folderze: {folder_path}')
    exit()

# 📏 Wczytaj pierwszy obraz by sprawdzić rozmiar
frame = cv2.imread(images[0])
height, width, _ = frame.shape

# 🎬 Parametry wideo
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = f'./frames_output/{os.path.splitext(nazwa_seq)[0]}.mp4'

# 🧩 Inicjalizacja zapisu wideo
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f'Tworzenie wideo z folderu: {folder_path}')
for img_path in images:
    img = cv2.imread(img_path)
    video.write(img)

video.release()
print(f'Gotowe! Wideo zapisane jako: {output_video}')

