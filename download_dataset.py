import kagglehub
import zipfile
import os

dataset_path = kagglehub.dataset_download("thecoderenroute/instagram-posts-dataset")

with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall("data")

print("✅ Veri seti başarıyla indirildi ve çıkarıldı.")
