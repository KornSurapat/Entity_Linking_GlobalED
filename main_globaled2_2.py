import unicodedata

import numpy as np
import seqeval.metrics
import spacy
import torch
from tqdm import tqdm, trange
from transformers import LukeTokenizer, LukeForEntitySpanClassification

# Load the model checkpoint
model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

# Load the tokenizer
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

# Recognizing named entities in a text
text = "Star Wars is a film written and directed by George Lucas"
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

entity_spans = []
original_word_spans = []
for token_start in doc:
    for token_end in doc[token_start.i:]:
        entity_spans.append((token_start.idx, token_end.idx + len(token_end)))
        original_word_spans.append((token_start.i, token_end.i + 1))

inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt", padding=True)
# inputs = inputs.to("cuda")
with torch.no_grad():
    outputs = model(**inputs)

print("outputs")
print(outputs)
print("----------" * 10)

logits = outputs.logits
max_logits, max_indices = logits[0].max(dim=1)

print("logits")
print(logits)
print("----------" * 10)
print("max_logits")
print(max_logits)
print("----------" * 10)
print("max_indices")
print(max_indices)
print("----------" * 10)

predictions = []
for logit, index, span in zip(max_logits, max_indices, original_word_spans):
    if index != 0:  # the span is not NIL
        predictions.append((logit, span, model.config.id2label[int(index)]))

print("predictions")
print(predictions)
print("----------" * 10)

# construct an IOB2 label sequence
# predicted_sequence = ["O"] * len(doc)
# for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
#     if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
#         predicted_sequence[span[0]] = "B-" + label
#         if span[1] - span[0] > 1:
#             predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

# for token, label in zip(doc, predicted_sequence):
#     print(token, label)