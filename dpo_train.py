import torch

from pathlib import Path
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


# Paths
BASE_MODEL_PATH = r"D:\models\Llama-3.2-1B-Instruct"

SFT_ADAPTER_PATH = (
    r"D:\ticket-priority-bot\models\ticket-priority-sft"
)

TRAIN_FILE = r"D:\ticket-priority-bot\data\dpo\train.jsonl"
VALIDATION_FILE = (
    r"D:\ticket-priority-bot\data\dpo\validation.jsonl"
)

OUTPUT_DIR = (
    r"D:\ticket-priority-bot\models\ticket-priority-dpo"
)


print("=" * 60)
print("TICKET PRIORITY BOT - DPO")
print("=" * 60)


# Check that everything we need is available.
print("\n[1/5] Checking files...")

required_paths = {
    "Base model": BASE_MODEL_PATH,
    "SFT adapter": SFT_ADAPTER_PATH,
    "DPO training dataset": TRAIN_FILE,
    "DPO validation dataset": VALIDATION_FILE,
}

for name, path in required_paths.items():
    if Path(path).exists():
        print(f"  [OK] {name}: {path}")
    else:
        print(f"  [ERROR] {name} not found: {path}")
        raise FileNotFoundError(path)


# Check the GPU before loading the model.
print("\n[2/5] Checking GPU...")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for DPO training.")

print("  [OK] CUDA available")
print(f"  GPU: {torch.cuda.get_device_name(0)}")

total_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"  VRAM: {total_memory:.2f} GB")


# Load the DPO datasets.
print("\n[3/5] Loading DPO datasets...")

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train",
)

validation_dataset = load_dataset(
    "json",
    data_files=VALIDATION_FILE,
    split="train",
)

print(
    f"  [OK] Training dataset loaded: "
    f"{len(train_dataset)} examples"
)

print(
    f"  [OK] Validation dataset loaded: "
    f"{len(validation_dataset)} examples"
)

print(f"  Columns: {train_dataset.column_names}")

print("\n  Sample DPO example:")
print(train_dataset[0])


# Load the original Llama model in 4-bit.
# We're not training the base model itself.
print("\n[4/5] Loading base Llama model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)

print("  [OK] Base model loaded")


# Load the tokenizer from the original model.
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("  [OK] Tokenizer loaded")


# Now load the adapter produced by our SFT run.
#
# This is the important part:
#
# Base Llama
#      +
# SFT adapter
#      ↓
# model used by DPO
#
# We are NOT creating another LoRA adapter here.
model = PeftModel.from_pretrained(
    model,
    SFT_ADAPTER_PATH,
    is_trainable=True,
)

print("  [OK] SFT adapter loaded")
print("  [OK] SFT adapter is trainable")


# Make sure the model knows we're doing training.
model.config.use_cache = False


# DPO configuration.
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    # Keep this small for the 6 GB GPU.
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    # Start with 2 epochs for DPO.
    num_train_epochs=2,

    # Lower than our SFT learning rate.
    learning_rate=5e-5,

    max_length=512,

    gradient_checkpointing=True,

    # Saves optimizer memory.
    optim="paged_adamw_8bit",

    # Controls how strongly DPO pushes the model
    # toward the chosen response.
    beta=0.1,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    logging_steps=10,

    report_to="none",
)


print("\n[5/5] Initializing DPOTrainer...")


# The SFT-trained PeftModel is passed directly here.
#
# We intentionally do NOT provide:
#
#     peft_config=...
#
# because the model already contains our SFT adapter.
trainer = DPOTrainer(
    model=model,

    # With a PEFT model, TRL can use the base model
    # with the adapter disabled as the reference model.
    ref_model=None,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    processing_class=tokenizer,
)


print("  [OK] DPOTrainer initialized")

print("\n" + "=" * 60)
print("Starting DPO training...")
print("=" * 60 + "\n")


trainer.train()


# Save the DPO-trained adapter.
#
# This contains the updated adapter weights after DPO.
trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


print("\n" + "=" * 60)
print("DPO training complete!")
print(f"Final adapter saved to:")
print(OUTPUT_DIR)
print("=" * 60)