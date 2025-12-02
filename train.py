import os
import torch
import config
from data_utils import get_train_val_datasets, formatting_func

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_model_and_tokenizer():
    print(f"Model yükleniyor → {config.MODEL_NAME}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # LoRA training için kapalı olmalı
    
    return model, tokenizer

def get_lora_model(model):
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None,  # burada q_proj,k_proj yada MLP için özel seçilebilir
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def main(): 
    
    # 1) Dataset
    # samples alanı ne kadar veri kullanılacağını belirtir, None -> hepsi
    train_ds, val_ds = get_train_val_datasets(
        dataset_kind=dataset_kind,
        train_ratio=0.95,
        max_train_samples=None   # ilk denemeler için küçük tutuldu 500 ile, 
    )

    # 2) Model + tokenizer
    model, tokenizer = get_model_and_tokenizer()
    model = get_lora_model(model)

    # 3) Tokenization
    def tokenize(example):
        text = formatting_func(example)
        return tokenizer(
            text,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            padding=False,
        )

    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # bf16'i güvenli kullanmak için küçük bir kontrol:
    use_bf16 = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:  # Ampere ve üstü
            use_bf16 = True

    training_args = TrainingArguments(
        output_dir=config.CKPT_DIR,
        run_name=config.RUN_NAME,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LR,
        bf16=use_bf16,
        logging_dir=config.LOG_DIR,
        logging_steps=config.LOG_STEPS,
        logging_first_step=True,
        report_to=["tensorboard"],

        eval_strategy="steps",  
        eval_steps=100,                
        eval_delay=0,
        eval_on_start=True,

        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,     # son iki seferdeki kaydediliyor

        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=config.WARMUP_RATIO,

        seed=config.SEED,
        data_seed=config.SEED,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    
    # hangi veri ile eğitilecekse onu çalıştırmak yeterli
    # DEEP
    main(dataset_kind="deep")

    # DIVERSE
    # main(dataset_kind="diverse")
