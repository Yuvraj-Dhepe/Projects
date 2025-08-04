from datasets import load_dataset

## Step 1: Load the SQuAD dataset
dataset = load_dataset("squad")

## Step 2 : Extract unique contexts from the dataset
data = [item["context"] for item in dataset["train"]]
texts = list(set(data))


