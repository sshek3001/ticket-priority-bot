import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


DATASET_NAME = "irongateprd/support-ticket-intents"

OUTPUT_DIR = Path("data")

random.seed(42)

PRIORITIES = ["P1", "P2", "P3", "P4"]


# For now we're using one project.
# Later we can add more companies with their own priority rules.
PROJECTS = {
    "Acme Payments": {
        "description": "Payment processing platform",
        "priority_rules": {
            "P1": (
                "Critical incidents affecting payment processing "
                "for many customers or causing a major service outage."
            ),
            "P2": (
                "Major issues affecting an individual customer's payment "
                "or an important payment workflow."
            ),
            "P3": (
                "Minor issues with a workaround available and limited "
                "customer impact."
            ),
            "P4": (
                "General requests, informational questions, or issues "
                "with minimal business impact."
            ),
        },
    }
}


def load_source_dataset():
    """
    The dataset's metadata currently has a ClassLabel issue, so instead
    of loading the whole Hugging Face dataset normally, we download the
    JSONL files directly.
    """

    splits = {}

    for split in ["train", "validation", "test"]:
        filename = f"data/{split}-00000-of-00001.jsonl"

        file_path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=filename,
            repo_type="dataset",
        )

        # Loading the JSONL through the generic JSON loader means we don't
        # depend on the dataset's broken metadata.
        splits[split] = load_dataset(
            "json",
            data_files=file_path,
            split="train",
        )

    return splits


def build_priority_policy(project_name):
    """Convert the project's priority rules into text for the model."""

    project = PROJECTS[project_name]

    rules = "\n".join(
        f"{priority}: {description}"
        for priority, description in project["priority_rules"].items()
    )

    return (
        f"Project: {project_name}\n"
        f"Project type: {project['description']}\n\n"
        f"Priority policy:\n{rules}"
    )


def build_ticket_prompt(example, project_name):
    """Create the prompt that the model will receive."""

    policy = build_priority_policy(project_name)

    subject = example["subject"].strip()
    body = example["body"].strip()

    return (
        "You are a ticket priority classification assistant.\n\n"
        f"{policy}\n\n"
        "Classify the following support ticket according to the "
        "project's priority policy.\n\n"
        f"Subject: {subject}\n"
        f"Ticket: {body}\n\n"
        "Output only one priority label: P1, P2, P3, or P4."
    )


def create_sft_example(example, project_name):
    """
    Create the format expected by SFTTrainer.

    The model learns:
        ticket + project policy -> correct priority
    """

    priority = example["priority"].strip().upper()

    if priority not in PRIORITIES:
        raise ValueError(f"Unexpected priority: {priority}")

    return {
        "messages": [
            {
                "role": "user",
                "content": build_ticket_prompt(
                    example,
                    project_name,
                ),
            },
            {
                "role": "assistant",
                "content": priority,
            },
        ]
    }


def create_dpo_example(example, project_name):
    """
    Create the format we'll eventually use for DPO.

    The correct priority is the preferred answer.
    A different priority is the rejected answer.
    """

    correct_priority = example["priority"].strip().upper()

    if correct_priority not in PRIORITIES:
        raise ValueError(f"Unexpected priority: {correct_priority}")

    wrong_priorities = [
        priority
        for priority in PRIORITIES
        if priority != correct_priority
    ]

    rejected_priority = random.choice(wrong_priorities)

    return {
        "prompt": build_ticket_prompt(
            example,
            project_name,
        ),
        "chosen": correct_priority,
        "rejected": rejected_priority,
    }


def save_jsonl(data, file_path):
    """Save a list of dictionaries as JSONL."""

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for example in data:
            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


def process_split(dataset, split_name, project_name):
    """Create both SFT and DPO data for one dataset split."""

    sft_data = []
    dpo_data = []

    for example in dataset:

        sft_data.append(
            create_sft_example(
                example,
                project_name,
            )
        )

        dpo_data.append(
            create_dpo_example(
                example,
                project_name,
            )
        )

    save_jsonl(
        sft_data,
        OUTPUT_DIR / "sft" / f"{split_name}.jsonl",
    )

    save_jsonl(
        dpo_data,
        OUTPUT_DIR / "dpo" / f"{split_name}.jsonl",
    )

    print(
        f"{split_name}: "
        f"{len(sft_data)} SFT examples, "
        f"{len(dpo_data)} DPO examples"
    )


def main():

    print(f"Loading {DATASET_NAME}...")

    dataset = load_source_dataset()

    print("\nDataset loaded successfully:")

    for split_name, split in dataset.items():
        print(f"  {split_name}: {len(split)} examples")

    project_name = "Acme Payments"

    print(f"\nCreating training data for: {project_name}")

    for split_name, split in dataset.items():

        process_split(
            split,
            split_name,
            project_name,
        )

    print("\nDone.")

    print("\nGenerated files:")

    for file_path in OUTPUT_DIR.rglob("*.jsonl"):
        print(f"  {file_path}")


if __name__ == "__main__":
    main()