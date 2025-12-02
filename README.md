## NLP Dersi, LoRA ile Fine-Tuning - Proje 2


### Lora eğitimi parametlerleri

* 20131MiB /  81920MiB  —> vram kullanımı, deep ve diverse için de aynı oldu. 
* 20209MiB /  81920MiB ——> 2. Denemedeki parametreler ile vram kullanımı.

DEEP ve DIVERSE parametreleri ;ilk eğitim denemelerinde her iki veri seti ile eğitim parametreleri değiştirilmeden aynı olarak kullanıldı. Tüm veri seti ile eğitimler yapıldı. 
* MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
* MAX_SEQ_LEN = 1024 # ilk önce 1024 idi çok yavaştı 20 saat civarı eğitim süresi, 
* BATCH_SIZE = 8 # ilk denemede 2 idi, 
* LR = 2e-4
* NUM_EPOCHS = 3  # ilk deneme 1 ile yapılmıştı fena değildi, 
* WARMUP_RATIO = 0.03
* LORA_R = 16
* LORA_ALPHA = 32
* LORA_DROPOUT = 0.05
* SAVE_STEPS = 200
* LOG_STEPS = 20
* SEED = 42
* train_ratio: float = 0.95,


DEEP ve DIVERSE 2. Deneme Parametreleri;
Yine her iki veri seti için de parametrelerin değerleri aynı tutularak eğitilecekler. Tüm veri seti kullanıldı. 
* MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
* MAX_SEQ_LEN = 1024
* BATCH_SIZE = 8
* GRAD_ACCUM_STEPS = 2
* LR = 1.5e-4
* NUM_EPOCHS = 4
* WARMUP_RATIO = 0.05
* LORA_R = 32
* LORA_ALPHA = 64
* LORA_DROPOUT = 0.08
* SAVE_STEPS = 200
* LOG_STEPS = 20
* SEED = 42
* train_ratio=0.95
