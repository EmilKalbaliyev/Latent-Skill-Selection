from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch
import gc
import os
import re
import json
import torch
import numpy as np
import random
from transformers import set_seed

def set_all_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    set_seed(42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def llm_init(model_name):
    global device
    global model
    global tokenizer

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)


def generate_llm_response( prompt, choices=["A", "B", "C", "D"], apply_chat_template = True):

    if apply_chat_template:
      messages = [
      {"role": "user", "content": prompt},
      ]
      inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(device)

    else:
      inputs = tokenizer(prompt, return_tensors="pt", return_dict=True).to(device)

    with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_new_tokens=1,
          return_dict_in_generate=True,
          output_scores=True,
          do_sample=False
      )

    next_token_logits = outputs.scores[0][0]

    choice_token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]

    choice_logits = torch.tensor(
        [next_token_logits[token_id].item() for token_id in choice_token_ids],
        device=next_token_logits.device
    )
    
    choice_probs_tensor = torch.softmax(choice_logits, dim=-1)

    choice_probs = {
        c.strip(): choice_probs_tensor[i].item()
        for i, c in enumerate(choices)
    }

    pred = max(choice_probs, key=choice_probs.get)

    return choice_probs, pred


def write_jsonl_line(filepath, data: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if data is None:
        data = {}
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Local json dataset path"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Language model from Hugging Face"
    )

    parser.add_argument(
        "--apply_chat_template",
        action=argparse.BooleanOptionalAction,
    )
    
    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        help="Folder name to store results"
    )

    args = parser.parse_args()

    dataset_name = args.dataset_path
    model_name = args.model_name
    apply_chat_template = args.apply_chat_template
    results_folder = args.results_folder
    set_all_seeds()

    llm_init(model_name)

    dataset = load_dataset("json", data_files=dataset_name)['train']
    choice_letters = ['A','B','C','D']


    for d in tqdm(dataset):
        result={ 'id':d['id'], 'correct_answer_id':d['correct_answer_id'] }

        base_prompt = "CONTEXT:\n" + d["context"] + "\n\n" + "QUESTION: " + d["question"] + "\nOPTIONS:"

        for index, value in enumerate(d["answers"]):
            base_prompt += ("\n " + choice_letters[index] + ". " + value)

        qa_prompt = "Read the context and choose the correct option to answer the question. Only respond with the correct option letter.\n\n" + base_prompt + "\nANSWER: " 
        result['qa_choice_probs'], result['qa_pred'] = generate_llm_response(qa_prompt, choice_letters, apply_chat_template)
        result['qa_pred_id'] = choice_letters.index(result['qa_pred'])
        result['qa_score']= 1 if result['qa_pred_id']==d['correct_answer_id'] else 0

        write_jsonl_line(f"{results_folder}/{model_name.split('/')[1]}/output.jsonl",result)

if __name__ == "__main__":
    main()
