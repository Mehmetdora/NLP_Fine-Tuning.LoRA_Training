from datasets import load_dataset
from typing import Dict
import config

# ödevde kullanılmak zorunda olunan eğitim promtu her train için
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Please read the problem carefully before writing any Python code."
)

def load_codegen_dataset(split: str = "train", dataset_kind: str = "deep"):
    
    if dataset_kind == "deep":
        dataset_name = config.DATASET_DEEP_NAME
    elif dataset_kind == "diverse":
        dataset_name = config.DATASET_DIVERSE_NAME
    else:
        raise ValueError(f"Geçersiz dataset_kind: {dataset_kind}. 'deep' veya 'diverse' olmalı.")

    # kütüphane aracılığı ile yükleniyor , ben özel olarak indirmiyorum yani
    ds = load_dataset(dataset_name)
    return ds[split]

def formatting_func(example: Dict) -> str:
    
    # ödevde sadece input ve solution verileri kullanılacağı için
    inp = example["input"]
    sol = example["solution"]

    #tek bir promt haline getiriliyor. 
    text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{inp.strip()}\n"
        f"<|assistant|>\n{sol.strip()}"
    )
    return text

def get_train_val_datasets(
    dataset_kind: str = "deep",
    train_ratio: float = 0.9,
    max_train_samples: int = None
):
    
    full_train = load_codegen_dataset("train", dataset_kind=dataset_kind)
    full_train = full_train.shuffle(seed=config.SEED)

    if max_train_samples is not None:
        full_train = full_train.select(range(max_train_samples))

    # toplam veri ve train için kullanılacak verinin ayrılması
    n_total = len(full_train)
    n_train = int(n_total * train_ratio)

    # validation verilerinin ayrılması
    train_ds = full_train.select(range(n_train))
    val_ds = full_train.select(range(n_train, n_total))

    print(f"[{dataset_kind.upper()}] Toplam train örneği: {n_total}")
    print(f"[{dataset_kind.upper()}] Train: {len(train_ds)}, Val: {len(val_ds)}")

    return train_ds, val_ds
