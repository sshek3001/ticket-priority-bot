import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from pathlib import Path

# Paths
MODEL_PATH = r"D:\models\Llama-3.2-1B-Instruct"

TRAIN_FILE = r"D:\ticket-priority-bot\data\sft\train.jsonl"
VALIDATION_FILE = r"D:\ticket-priority-bot\data\sft\validation.jsonl"

OUTPUT_DIR = r"D:\ticket-priority-bot\models\ticket-priority-sft"


print("=" * 60)
print("TICKET PRIORITY BOT - QLoRA SFT")
print("=" * 60)


# Check the important files before doing anything else.
print("\n[1/5] Checking files...")

required_files = {
    "Model directory": MODEL_PATH,
    "Training dataset": TRAIN_FILE,
    "Validation dataset": VALIDATION_FILE,
}

for name, path in required_files.items():
    if Path(path).exists():
        print(f"  [OK] {name}: {path}")
    else:
        print(f"  [ERROR] {name} not found: {path}")
        raise FileNotFoundError(path)


# Check CUDA because we want to train on the GPU.
print("\n[2/5] Checking GPU...")

if torch.cuda.is_available():
    print(f"  [OK] CUDA available")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"  VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
else:
    print("  [WARNING] CUDA is not available.")
    print("  Training will not be practical on CPU.")


# Load the datasets we generated earlier.
print("\n[3/5] Loading datasets...")

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

print(f"  [OK] Training dataset loaded: {len(train_dataset)} examples")
print(
    f"  [OK] Validation dataset loaded: "
    f"{len(validation_dataset)} examples"
)

# Print one example so we can immediately verify the format.
print("\n  Sample SFT example:")
print(train_dataset[0])


# Load the model in 4-bit.
# This is the Q in QLoRA.
print("\n[4/5] Loading Llama 3.2 1B-Instruct in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)

print("  [OK] Model loaded")


# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("  [OK] Tokenizer loaded")


# LoRA adds a small number of trainable parameters while keeping
# the original Llama weights frozen.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


# Training configuration.
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    learning_rate=2e-4,

    max_length=512,

    gradient_checkpointing=True,

    optim="paged_adamw_8bit",

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    logging_steps=10,

    report_to="none",
)


# Put everything together.
print("\n[5/5] Initializing SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("  [OK] SFTTrainer initialized")
print("\n" + "=" * 60)
print("Starting QLoRA SFT training...")
print("=" * 60 + "\n")


trainer.train()


# Save the LoRA adapter.
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print("Training complete!")
print(f"Adapter saved to: {OUTPUT_DIR}")
print("=" * 60)