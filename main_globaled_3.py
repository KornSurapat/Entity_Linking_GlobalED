import json
import torch
from tqdm import trange
from transformers import LukeTokenizer, LukeForEntityClassification

# Load the model checkpoint
model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

# Load the tokenizer
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

# Detecting types of entities in a text
text = "Beyoncé lives in Los Angeles."
entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"

inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
outputs = model(**inputs)

predicted_indices = [index for index, logit in enumerate(outputs.logits[0]) if logit > 0]
print("Predicted entity type for Beyoncé:", [model.config.id2label[index] for index in predicted_indices])

entity_spans = [(17, 28)]  # character-based entity span corresponding to "Beyoncé"
inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
inputs.to("cuda")
outputs = model(**inputs)

predicted_indices = [index for index, logit in enumerate(outputs.logits[0]) if logit > 0]
print("Predicted entity type for Los Angeles:", [model.config.id2label[index] for index in predicted_indices])