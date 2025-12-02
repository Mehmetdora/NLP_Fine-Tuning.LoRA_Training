import os
from datetime import datetime

# 1) RUN ADI
RUN_TAG = "deep_full_e3"   # eğitimin log dosyasının ve checkpoint klasörünün ismi olacak, her eğitim için özel bir isim ver
RUN_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{RUN_TIME}_{RUN_TAG}"

# Ana proje dizini, bunu colabda açtıktan sonra projenin yolunu kendine göre düzenle
PROJECT_DIR = "/content/drive/MyDrive/lora-nlp-homework1"

# 2) Log ve eğitim sonu dosyalarının kaydedileceği klasörleri belirle
CHECKPOINT_ROOT = os.path.join(PROJECT_DIR, "checkpoints")
LOG_ROOT = os.path.join(PROJECT_DIR, "logs")

# 3) Her eğitim için farklı bir klasör oluşturulması için 
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, RUN_NAME)
LOG_DIR = os.path.join(LOG_ROOT, RUN_NAME)

# eğer klasörler yoksa oluştur
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("RUN_NAME:", RUN_NAME)
print("CKPT_DIR:", CKPT_DIR)
print("LOG_DIR:", LOG_DIR)

# kullanılacak model id'si
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# veriseti isimleri tanımlamaları
DATASET_DEEP_NAME = "Naholav/CodeGen-Deep-5K"
DATASET_DIVERSE_NAME = "Naholav/CodeGen-Diverse-5K"

# Eğitim ayarları -  bu değerler gpu imkanına göre ve amaca göre düzenlenmeli tekrar
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4 # ilk denemede 2 idi, 
GRAD_ACCUM_STEPS = 8    
LR = 2e-4
NUM_EPOCHS = 3  # ilk deneme 1 ile yapılmıştı fena değildi, 
WARMUP_RATIO = 0.03

# LoRA ayarları
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Kaydetme / log
SAVE_STEPS = 50
LOG_STEPS = 20

# Seed
SEED = 42
print("Config yüklendi. PROJECT_DIR:", PROJECT_DIR)
