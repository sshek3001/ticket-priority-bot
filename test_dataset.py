from pathlib import Path

from datasets import load_dataset


# These are the exact files we're trying to load.
TRAIN_FILE = r"D:\ticket-priority-bot\data\sft\train.jsonl"
VALIDATION_FILE = r"D:\ticket-priority-bot\data\sft\validation.jsonl"


print("=" * 60)
print("TESTING SFT DATASET LOADING")
print("=" * 60)


# First make sure the files actually exist.
print("\nChecking files...")

for name, path in [
    ("Train", TRAIN_FILE),
    ("Validation", VALIDATION_FILE),
]:
    if Path(path).exists():
        print(f"[OK] {name}: {path}")
    else:
        print(f"[ERROR] {name} file not found: {path}")
        raise FileNotFoundError(path)


# Try loading the training file.
print("\nLoading training dataset...")

try:
    train_dataset = load_dataset(
        "json",
        data_files=TRAIN_FILE,
        split="train",
    )

    print("[OK] Training dataset loaded")
    print(f"Number of examples: {len(train_dataset)}")
    print(f"Columns: {train_dataset.column_names}")

except Exception as e:
    print("[ERROR] Failed to load training dataset")
    print(type(e).__name__)
    print(e)
    raise


# Try loading the validation file separately.
print("\nLoading validation dataset...")

try:
    validation_dataset = load_dataset(
        "json",
        data_files=VALIDATION_FILE,
        split="train",
    )

    print("[OK] Validation dataset loaded")
    print(f"Number of examples: {len(validation_dataset)}")
    print(f"Columns: {validation_dataset.column_names}")

except Exception as e:
    print("[ERROR] Failed to load validation dataset")
    print(type(e).__name__)
    print(e)
    raise


# Print an actual example so we can verify that the JSON
# has the structure we expect.
print("\n" + "=" * 60)
print("SAMPLE TRAINING EXAMPLE")
print("=" * 60)

print(train_dataset[0])


print("\n" + "=" * 60)
print("DATASET TEST PASSED")
print("=" * 60)