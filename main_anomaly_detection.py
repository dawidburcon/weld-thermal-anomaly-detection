import os
import cv2
import torch
import numpy as np
from datetime import datetime
from torchvision import transforms
from weld_model import ConvAutoencoder
from PIL import Image
import logging
import matplotlib.pyplot as plt
import pandas as pd
import re


class AnomalyDetector:
    def __init__(self, config):
        self.config = config
        self._setup_paths()
        self._setup_logger()
        self._load_model()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def _setup_paths(self):
        for subdir in ["preview", "processed", "csv"]:
            os.makedirs(os.path.join(self.config['log_dir'], subdir), exist_ok=True)

    def _setup_logger(self):
        log_file = os.path.join(self.config['log_dir'], "system.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file, "a"), logging.StreamHandler()]
        )
        self.logger = logging.getLogger("AnomalyDetector")

    def _load_model(self):
        with open(self.config['threshold_path'], "r") as f:
            self.threshold = float(f.read().strip())

        self.model = ConvAutoencoder().to(self.config['device'])
        self.model.load_state_dict(torch.load(self.config['model_path'], map_location=self.config['device']))
        self.model.eval()

    def _max_difference(self, a, b):
        return torch.abs(a - b).max().item()

    def _analyze_roi(self, roi_gray):
        roi_f32 = roi_gray.astype(np.float32)
        mu, sigma = roi_f32.mean(), roi_f32.std()
        temp_mask = ((roi_f32 > mu + 3.5 * sigma) | (roi_f32 < mu - 3.5 * sigma)).astype(np.uint8) * 255
        cnts_temp, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_temp = [c for c in cnts_temp if cv2.contourArea(c) > self.config['min_contour_area_term']]
        num_temp = len(cnts_temp)
        ratio_tmp = (temp_mask == 255).sum() / temp_mask.size * 100
        area_mean = sum(cv2.contourArea(c) for c in cnts_temp) / num_temp if num_temp else 0
        is_term_bad = (ratio_tmp > self.config['term_thresh_percent']) or (area_mean > self.config['term_thresh_area'])

        roi_edges = cv2.Canny(cv2.GaussianBlur(roi_gray, (3, 3), 0), 50, 120)
        cnts_geom, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_geom = [c for c in cnts_geom if cv2.contourArea(c) > self.config['min_contour_area_geom']]
        num_geom = len(cnts_geom)
        area_geom = sum(cv2.contourArea(c) for c in cnts_geom)
        is_geom_bad = (num_geom > self.config['geom_thresh_contours']) or (area_geom > self.config['geom_thresh_area'])

        if is_term_bad and is_geom_bad:
            return "MIESZANA", roi_edges, num_geom, area_geom, num_temp, ratio_tmp
        elif is_term_bad:
            return "TERM_WADA", roi_edges, num_geom, area_geom, num_temp, ratio_tmp
        elif is_geom_bad:
            return "GEOM_WADA", roi_edges, num_geom, area_geom, num_temp, ratio_tmp
        else:
            return "POTWIERDZONA_AE", roi_edges, num_geom, area_geom, num_temp, ratio_tmp

    def run(self):
        for fname in sorted(os.listdir(self.config['image_dir'])):
            if not fname.lower().endswith((".jpg", ".png")):
                continue

            image_path = os.path.join(self.config['image_dir'], fname)
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"Nie udało się wczytać pliku: {fname}")
                continue

            roi = image[self.config['roi'][1]:self.config['roi'][3], self.config['roi'][0]:self.config['roi'][2]]
            roi_tensor = self.transform(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.config['device'])

            with torch.no_grad():
                recon = self.model(roi_tensor)
            diff_val = self._max_difference(roi_tensor, recon)

            if diff_val <= self.threshold:
                self.logger.info(f"frame={fname} | reconstruction_diff={diff_val:.6f} | anomaly=False")
                continue

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            anomaly_type, roi_edges, num_geom, area_geom, num_temp, ratio_tmp = self._analyze_roi(roi_gray)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{os.path.splitext(fname)[0]}_{ts}"

            cv2.imwrite(os.path.join(self.config['log_dir'], "preview", f"{base}.jpg"), image)
            cv2.imwrite(os.path.join(self.config['log_dir'], "processed", f"{base}_edges.jpg"), roi_edges)
            np.savetxt(os.path.join(self.config['log_dir'], "csv", f"{base}.csv"), roi_gray, fmt='%d', delimiter=',')

            self.logger.info(
                f"frame={fname} | reconstruction_diff={diff_val:.6f} | anomaly=True | type={anomaly_type} "
                f"| geom_cnt={num_geom} geom_area={area_geom:.1f} | temp_cnt={num_temp} temp_ratio={ratio_tmp:.2f}%"
            )

        self.logger.info("Zakonczono analizę wszystkich klatek.")
        self._generate_plot()

    def _generate_plot(self):
            log_path = os.path.join(self.config['log_dir'], "system.log")
            pattern = r"frame=(.+?) \| reconstruction_diff=([\d.]+) \| anomaly=(True|False)(?: \| type=(\w+))?"
            data = []

            with open(log_path, "r") as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        frame, diff, is_anom, typ = match.groups()
                        is_anomaly = is_anom == "True"
                        typ = typ if typ else "OK"
                        final_type = typ
                        if is_anomaly and typ == "OK":
                            final_type = "FALSE_POSITIVE"
                        data.append({
                            "frame": frame,
                            "recon_error": float(diff),
                            "anomaly": is_anomaly,
                            "type": final_type
                        })

            if not data:
                self.logger.warning("Brak danych do wykresu.")
                return

            df = pd.DataFrame(data)

            # === STATYSTYKI ===
            total = len(df)
            false_positives = (df["type"] == "FALSE_POSITIVE").sum()
            confirmed = df["type"].isin(["TERM_WADA", "GEOM_WADA", "MIESZANA", "POTWIERDZONA_AE"]).sum()
            self.logger.info(f"📊 Statystyki:")
            self.logger.info(f" - Wszystkie klatki: {total}")
            self.logger.info(f" - Potwierdzone anomalie: {confirmed}")
            self.logger.info(f" - Fałszywe anomalie (FALSE_POSITIVE): {false_positives}")

            # === WYKRES ===
            plt.figure(figsize=(14, 5))
            plt.plot(df.index, df["recon_error"], label="Błąd rekonstrukcji", color="black")

            for anomaly_type, color in {
                "TERM_WADA": "orange",
                "GEOM_WADA": "blue",
                "MIESZANA": "red",
                "POTWIERDZONA_AE": "purple",
                "FALSE_POSITIVE": "gray"
            }.items():
                subset = df[df["type"] == anomaly_type]
                if not subset.empty:
                    plt.scatter(subset.index, subset["recon_error"], label=anomaly_type, color=color)

            plt.axhline(self.threshold, linestyle='--', color='gray', label="Próg AE")
            plt.xlabel("Numer próbki")
            plt.ylabel("Błąd rekonstrukcji")
            plt.title("Błąd rekonstrukcji i typy anomalii")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_path = os.path.join(self.config['log_dir'], "reconstruction_error_plot.png")
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"📈 Zapisano wykres do: {plot_path}")


if __name__ == "__main__":
    config = {
        'model_path': "models/weld_autoencoder_test.pth",
        'threshold_path': "models/weld_threshold.txt",
        # 'image_dir': "frames_output/625_38n18_1_2mm_-161_07_41_19_806/preview_fixed",
        'image_dir': "frames_output/600_56n17_1mm_-161_09_29_59_808/preview_fixed",
        'log_dir': "logs/anomalies",
        'roi': (230, 0, 345, 420),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'geom_thresh_contours': 3,
        'geom_thresh_area': 50,
        'term_thresh_percent': 2.0,
        'term_thresh_area': 30,
        'min_contour_area_geom': 10,
        'min_contour_area_term': 5
    }

    detector = AnomalyDetector(config)
    detector.run()