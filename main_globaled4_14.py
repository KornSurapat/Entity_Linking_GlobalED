# import click
import logging
import os
import json
from typing import List, Optional

import dataclasses
# import nltk
# nltk.download('punkt')
import pywikibot
from pywikibot import exceptions

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from luke.utils.entity_vocab import EntityVocab, MASK_TOKEN, PAD_TOKEN

from dataloader_2 import create_dataloader
from dataset import load_dataset
from model import LukeForEntityDisambiguation

logger = logging.getLogger(__name__)

# @click.command()
# @click.option("--model-dir", type=click.Path(exists=True), required=True)
# @click.option("--device", type=str, default="cuda")
# @click.option("--max-seq-length", type=int, default=512)
# @click.option("--max-entity-length", type=int, default=128)
# @click.option("--max-candidate-length", type=int, default=30)
# @click.option("--max-mention-length", type=int, default=30)
# @click.option(
#     "--inference-mode", type=click.Choice(["global", "local"]), default="global",
# )
# @click.option(
#     "--document-split-mode", type=click.Choice(["simple", "per_mention"]), default="simple",
# )


@dataclasses.dataclass
class Candidate:
    title: str
    prior_prob: float

@dataclasses.dataclass
class Mention:
    text: str
    title: str
    index: int
    candidates: List[Candidate]
    start: Optional[int] = None
    end: Optional[int] = None

@dataclasses.dataclass
class Document:
    id: str
    text: str
    mentions: List[Mention]


# def find_words_index_by_text_index(text, words, ref_st_idx, ref_en_idx):
#     cur_idx = 0
#     cur_idx_2 = 0
#     cur_idx_3 = 0
#     for i in range(len(words)):
#         st_idx = text.find(words[i], cur_idx)
#         en_idx = st_idx + len(words[i])
#         if st_idx != -1:
#             cur_idx_3 = cur_idx_2
#             cur_idx_2 = cur_idx
#             cur_idx = en_idx
#             if st_idx == ref_st_idx:
#                 if en_idx >= ref_en_idx:
#                     return i, i + 1
#                 else:
#                     j = min(i + 1, len(words) - 1)
#                     while j < len(words) and en_idx < ref_en_idx:
#                         if text.find(words[j], en_idx) != -1: en_idx = text.find(words[j], en_idx) + len(words[j])
#                         j += 1
#                     return i, j
#             elif st_idx > ref_st_idx:
#                 i = max(i - 1, 0)
#                 while True:
#                     st_idx = text.find(words[i], cur_idx_3)
#                     en_idx = st_idx + len(words[i])
#                     if st_idx != -1 or i == 0: break
#                     i -= 1
#                 if en_idx >= ref_en_idx:
#                     return i, i + 1
#                 else:
#                     j = min(i + 1, len(words) - 1)
#                     while j < len(words) and en_idx < ref_en_idx:
#                         if text.find(words[j], en_idx) != -1: en_idx = text.find(words[j], en_idx) + len(words[j])
#                         j += 1
#                     return i, j
#             elif i == len(words) - 1 and words[i].find(text[ref_st_idx: ref_en_idx]) != -1:
#                 return i, i + 1
#     return -1, -1


qcode_to_title = {}
with open("qcode_to_title.txt", "r", encoding="utf-8") as file:
    for line in file:
        qcode_and_title = line.split(" -> ")
        qcode_to_title[qcode_and_title[0]] = qcode_and_title[1][:-1]


def get_wikipedia_title(wikidata_id):
    try:
        site = pywikibot.Site("wikidata", "wikidata")
        item = pywikibot.ItemPage(site, wikidata_id)
        sitelinks = item.sitelinks
        if "enwiki" in sitelinks:
            return sitelinks['enwiki'].title
        else:
            return None
    except pywikibot.exceptions.NoPageError:
        return None
    except pywikibot.exceptions.IsRedirectPageError:
        if not item.exists():
            return None
        sitelinks = item.sitelinks
        if "enwiki" in sitelinks:
            return sitelinks['enwiki'].title
        else:
            return None
    except Exception as e:
        return None


def evaluate(
    model_dir: str,
    device: str,
    max_seq_length: int,
    max_entity_length: int,
    max_candidate_length: int,
    max_mention_length: int,
    inference_mode: str,
    document_split_mode: str,
):
    model = LukeForEntityDisambiguation.from_pretrained(model_dir).eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    entity_vocab_path = os.path.join(model_dir, "entity_vocab.jsonl")
    entity_vocab = EntityVocab(entity_vocab_path)

    entity_vocab_list = []
    for entity in entity_vocab.vocab:
        entity_vocab_list.append(entity)

    pad_entity_id = entity_vocab[PAD_TOKEN]
    mask_entity_id = entity_vocab[MASK_TOKEN]

    test_set = ["ace2004", "aida", "aquaint", "cweb", "graphq", "mintaka", "msnbc", "reddit_comments", "reddit_posts", "shadow", "tail", "top", "tweeki", "webqsp", "wiki"]
    # test_set = ["ace2004", "aida", "aquaint", "cweb"]
    for dataset_name in test_set:
        print(f"========== Dataset: {dataset_name} ==========")

        test_data = []
        with open(f'ED_Test_Datasets/{dataset_name}.jsonl', 'r', encoding='utf-8') as f:
            for l in f:
                json_object = json.loads(l.strip())
                test_data.append(json_object)
        
        documents = []
        for d in range(len(test_data)):
            text = test_data[d]["text"]
            # words = nltk.word_tokenize(text)
            mentions = []
            for m in range(len(test_data[d]["gold_spans"])):
                # start_words_idx, end_words_idx = find_words_index_by_text_index(text,
                #                                                                 words,
                #                                                                 test_data[d]["gold_spans"][m]["start"],
                #                                                                 test_data[d]["gold_spans"][m]["start"] + test_data[d]["gold_spans"][m]["length"])
                
                if test_data[d]["gold_spans"][m]["wikidata_qid"] in qcode_to_title:
                    title = qcode_to_title[test_data[d]["gold_spans"][m]["wikidata_qid"]]
                else:
                    title = get_wikipedia_title(test_data[d]["gold_spans"][m]["wikidata_qid"])
                    qcode_to_title[test_data[d]["gold_spans"][m]["wikidata_qid"]] = title
                
                candidates = []
                for c in range(len(test_data[d]["gold_spans"][m]["candidates"])):
                    if test_data[d]["gold_spans"][m]["candidates"][c][0] in qcode_to_title:
                        t = qcode_to_title[test_data[d]["gold_spans"][m]["candidates"][c][0]]
                    else:
                        t = get_wikipedia_title(test_data[d]["gold_spans"][m]["candidates"][c][0])
                        qcode_to_title[test_data[d]["gold_spans"][m]["candidates"][c][0]] = t
                    candidate = Candidate(title=t, prior_prob=test_data[d]["gold_spans"][m]["candidates"][c][1])
                    candidates.append(candidate)
                
                mention = Mention(text=test_data[d]["text"][test_data[d]["gold_spans"][m]["start"]: 
                                       test_data[d]["gold_spans"][m]["start"] + test_data[d]["gold_spans"][m]["length"]], 
                                  title=title, 
                                  index=m, 
                                  candidates=candidates, 
                                  start=test_data[d]["gold_spans"][m]["start"], 
                                  end=test_data[d]["gold_spans"][m]["start"] + test_data[d]["gold_spans"][m]["length"])
                mentions.append(mention)
            
            document = Document(id="", text=text, mentions=mentions)
            documents.append(document)
        
        ###
        # print("type(documents)")
        # print(type(documents))
        # print("documents[0]")
        # print(documents[0])
        # print("-" * 90)
        ###
        
        dataloader = create_dataloader(
            documents=documents,
            tokenizer=tokenizer,
            entity_vocab=entity_vocab,
            batch_size=1,
            fold="eval",
            document_split_mode=document_split_mode,
            max_seq_length=max_seq_length,
            max_entity_length=max_entity_length,
            max_candidate_length=max_candidate_length,
            max_mention_length=max_mention_length,
        )

        candidate_indices_list = []
        eval_entity_mask_list = []
        scores_list = []
        
        ###
        # i = 0
        ###

        for input_dict in tqdm(dataloader, leave=False):
            inputs = {k: v.to(device) for k, v in input_dict.items()}
            # inputs = {k: v for k, v in input_dict.items()}

            ###
            # if i == 0 or i == 1:
            #     for k, v in input_dict.items():
            #         print(k)
            #         print(v)
            #         print("-" * 60)
            #     print("-" * 90)
            ###

            entity_ids = inputs.pop("entity_ids")
            entity_length = inputs["entity_attention_mask"].sum()
            input_entity_ids = entity_ids.new_full(entity_ids.size(), pad_entity_id)
            input_entity_ids[0, :entity_length] = mask_entity_id
            eval_entity_mask = inputs.pop("eval_entity_mask")
            eval_entity_mask_list.append(eval_entity_mask[0, :entity_length])
            
            with torch.no_grad():
                candidate_indices = torch.zeros(entity_length, 30, dtype=torch.long, device=device)
                scores = torch.zeros(entity_length, 30, dtype=torch.float, device=device)
                # candidate_indices = torch.zeros(entity_length, 30, dtype=torch.long)
                # scores = torch.zeros(entity_length, 30, dtype=torch.float)
                
                if inference_mode == "local":
                    logits = model(entity_ids=input_entity_ids, **inputs)[0]
                    for n, entity_id in enumerate(torch.argmax(logits, dim=2)[0, :entity_length]):
                        if inputs["entity_candidate_ids"][0, n].sum() != 0:
                            candidate_indices[n] = (inputs["entity_candidate_ids"][0, n] == entity_id).nonzero(
                                as_tuple=True
                            )[0][0]
                
                else:
                    for j in range(entity_length):
                        logits = model(entity_ids=input_entity_ids, **inputs)[0]
                        
                        probs = torch.nn.functional.softmax(logits, dim=2) * (
                            input_entity_ids == mask_entity_id
                        ).unsqueeze(-1).type_as(logits)

                        ###
                        # max_probs, max_indices = torch.max(probs.squeeze(0), dim=1)
                        ###
                        
                        top_k_probs = []
                        top_k_indices = []
                        max_k_probs, max_k_indices = torch.topk(probs.squeeze(0), k=30, dim=1)
                        for k in range(30):
                            top_k_probs.append(max_k_probs[:, k])
                            top_k_indices.append(max_k_indices[:, k])
                        
                        ###
                        # if i == 0 or i == 1:
                        #     print("max_probs")
                        #     print(max_probs)
                        #     print("max_indices")
                        #     print(max_indices)
                        #     print("top_k_probs")
                        #     print(top_k_probs)
                        #     print("top_k_indices")
                        #     print(top_k_indices)
                        #     print("-" * 60)
                        ###

                        target_index = torch.argmax(top_k_probs[0], dim=0)
                        input_entity_ids[0, target_index] = top_k_indices[0][target_index]
                            
                        for i in range(30):
                            candidate_indices[target_index, i] = top_k_indices[i][target_index]
                            scores[target_index, i] = top_k_probs[i][target_index]

                    ###
                    # if i == 0 or i == 1:
                    #     print("-" * 90)
                    ### 
                    
            candidate_indices_list.append(candidate_indices)
            scores_list.append(scores)

            ###
            # i += 1
            ###
        
        all_candidate_indices = torch.cat(candidate_indices_list)
        all_eval_entity_mask = torch.cat(eval_entity_mask_list)
        all_scores = torch.cat(scores_list)

        last_index = -1
        output = []
        
        for d, document in enumerate(documents):
            problem = {}
            problem["text"] = test_data[d]["text"]
            problem["text_2"] = document.text
            problem["gold_spans"] = []
            
            for m, mention in enumerate(document.mentions):
                mention_info = {}
                mention_info["start_text"] = test_data[d]["gold_spans"][m]["start"]
                mention_info["end_text"] = test_data[d]["gold_spans"][m]["start"] + test_data[d]["gold_spans"][m]["length"]
                mention_info["start_text_2"] = mention.start
                mention_info["end_text_2"] = mention.end
                
                index = last_index + 1
                while True:
                    if all_eval_entity_mask[index] == 1:
                        break
                    index += 1
                last_index = index
                
                predictions_globaled = []
                for i in range(30): 
                    predicted_index = all_candidate_indices[index][i]
                    predicted_title = entity_vocab_list[predicted_index].title
                    predicted_score = all_scores[index][i]
                    predictions_globaled.append([predicted_title, float(predicted_score)])
                mention_info["predictions_globaled"] = predictions_globaled 
                problem["gold_spans"].append(mention_info)
            
            output.append(problem)

        if dataset_name == "wiki":
            print("Just Test")
            print(output[0]["text"])
            print()
            print(output[0]["text_2"])
            print()
            for m in range(len(output[0]["gold_spans"])):
                print(output[0]["text"][output[0]["gold_spans"][m]["start_text"]: output[0]["gold_spans"][m]["end_text"]])
                print(output[0]["text_2"][output[0]["gold_spans"][m]["start_text_2"]: output[0]["gold_spans"][m]["end_text_2"]])
                print()
                print(output[0]["gold_spans"][m]["predictions_globaled"])
                print("-" * 30)
            print("-" * 60)
            # print(output[8]["text"])
            # print()
            # for m in range(len(output[8]["gold_spans"])):
            #     print(output[8]["text"][output[8]["gold_spans"][m]["start_text"]: output[8]["gold_spans"][m]["end_text"]])
            #     print(" ".join(output[8]["words"][output[8]["gold_spans"][m]["start_words"]: output[8]["gold_spans"][m]["end_words"]]))
            #     print()
            #     print(output[8]["gold_spans"][m]["predictions_globaled"])
            #     print("-" * 30)

        with open(f"ED_Test_Datasets_Pred/{dataset_name}_pred.jsonl", 'w', encoding='utf-8') as f:
            for item in output:
                f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    evaluate("luke_ed_large/luke_ed_large", "cuda", 512, 128, 30, 30, "global", "per_mention")