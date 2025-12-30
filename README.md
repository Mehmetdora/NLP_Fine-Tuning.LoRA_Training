# NLP Dersi: LoRA ile Fine-Tuning Projesi (Proje 2)

Bu proje, **Qwen2.5-Coder-1.5B** modeli üzerinde LoRA (Low-Rank Adaptation) yöntemi kullanılarak gerçekleştirilen bir ince ayar (fine-tuning) çalışmasını ve bu süreçteki teknik çıkarımları içermektedir.

## Proje Özeti

* **Base Model:** `Qwen2.5-Coder-1.5B-Instruct` (1.5B parametre)
* **Yöntem:** LoRA Fine-Tuning
* **Veri Setleri (Datasets):**
    * `Naholav/CodeGen-Diverse-5K`
    * `Naholav/CodeGen-Deep-5K`
* **Eğitim Alanları (Training Fields):** `input`, `solution` (Sadece kod odaklı, reasoning/muhakeme içermeyen yapı)


## 🛠️ Kullanılan Hiperparametreler ve Öğrendiklerim

### 1. Max Sequence Length (`max_seq_len`)
* **İşlevi:** Modelin "attention" matrisinin boyutunu ve tek seferde okuyabileceği metin uzunluğunu belirler (Context Window).
* **Etkisi:** Modelin aynı anda kaç token görebileceğini ve bağlam kurabileceğini tanımlar.
* **Donanım İlişkisi:** Bu parametrenin değeri **VRAM** kullanımı ile doğru orantılıdır. Ayrıca verilecek eğitim verisinin uzunluğu ile uyumlu olmalıdır.

### 2. Learning Rate (`lr`)
* **İşlevi:** Modelin her adımda ağırlıkları ne kadar değiştireceğini belirler. LoRA matrislerinin (A ve B) güncellenme hızıdır.
* **Denge:**
    * **Yüksek LR:** Model hızlı öğrenir ancak kararsız (unstable) hale gelebilir ve çıktıları bozulabilir.
    * **Düşük LR:** Model daha istikrarlı öğrenir, genel performans artar ve *overfit* riski azalır; ancak eğitim süresi çok uzayabilir.
    * **Özetle:** Çok büyük olursa model bozulur, çok küçük olursa model öğrenemez.

### 3. Rank (`r`)
* **İşlevi:** LoRA'nın eklediği matrislerin boyutunu belirler. Bilginin ne kadar detaylı kodlanacağını temsil eder.
* **Etkisi:**
    * **Büyük R:** LoRA daha fazla bilgiyi encode eder, ana model daha fazla değişime uğrar.
    * **Küçük R:** Model üzerinde çok sınırlı değişiklik yapar.
* **Donanım İlişkisi:** **VRAM** kullanımı ile birebir ilişkilidir. R değeri büyüdükçe parametre sayısı artacağı için eğitim hızı yavaşlar, azaldıkça hız artar.

### 4. Alpha (`lora_alpha`)
* **İşlevi:** LoRA güncellemelerinin temel modele ne kadarlık bir ölçekte etki edeceğini belirleyen katsayıdır (Güncelleme güç seviyesi).
* **Denge:**
    * **Büyük Alpha:** Model daha hızlı öğrenir ancak *overfitting* riski artar.
    * **Küçük Alpha:** *Overfitting* riski azalır ancak model domain bilgisini (yeni veriyi) yeterince öğrenemeyebilir.

### 5. Checkpoints
* **İşlevi:** Eğitim sırasında belirli adımlarda modelin LoRA ağırlıklarının ve optimizer durumunun kaydedilmesidir.
* **Avantajı:** Bir versiyonlama sistemi gibi çalışır. GPU kesintisi veya teknik aksaklıklarda eğitimin kaybedilmemesini (yedekleme) sağlar ve en iyi performans veren adımın seçilmesine olanak tanır.

### 6. Epoch
* **İşlevi:** Eğitim verilerinin tamamının modelden kaç kez geçirileceğini ifade eder.
* **Örnek:** 10.000 verilik bir set ve 10 epoch için toplamda 100.000 training step gerçekleşir.
* **Etkisi:**
    * **Fazla Epoch:** Model veriyi ezberlemeye başlar (*overfitting*).
    * **Az Epoch:** Model veriyi tam öğrenemez (*underfitting*).
* **Not:** Epoch sayısı eğitim süresini doğrudan etkiler ancak VRAM kullanımını etkilemez.
