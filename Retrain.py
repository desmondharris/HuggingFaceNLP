from datasets import load_dataset
from transformers import AutoTokenizer


cola_data = load_dataset("glue", "cola")
dataset = cola_data["train"]
