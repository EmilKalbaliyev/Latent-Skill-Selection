from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm
import argparse

def input_prompt(d):
  choice_letters = ['A','B','C','D']
  base_prompt = "CONTEXT:\n" + d["context"] + "\n\n" + "QUESTION: " + d["question"] + "\nOPTIONS:"

  for index, value in enumerate(d["answers"]):
      base_prompt += ("\n " + choice_letters[index] + ". " + value)

  prompt = "Read the context and choose the correct option to answer the question. Only respond with the correct option letter.\n\n" + base_prompt + "\nANSWER: "
  d['messages'] = [
      {"role": "user", "content": prompt},
      ]
  return d


def get_hidden_states(texts,model,tokenizer,device):
    all_layer_reps = []

    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer.apply_chat_template(text,
                                                   return_tensors="pt",
                                                   add_generation_prompt=True,
                                                   return_dict=True).to(device)
            outputs = model(**inputs, output_hidden_states=True,)

            hidden_states = outputs.hidden_states  

            layer_reps = []
            for layer in hidden_states:
                last_token = layer[:, -1, :].squeeze(0).cpu()
                layer_reps.append(last_token)

            all_layer_reps.append(layer_reps)
            del inputs

    return torch.stack([torch.stack(x) for x in all_layer_reps])

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Language model from Hugging Face"
    )


    args = parser.parse_args()
    model_name = args.model_name
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    dataset = load_dataset(
        'json',
        data_files={
            'train': '../dataset/train.json',
            'validation': '../dataset/validation.json',
            'challenge': '../dataset/challenge.json',
        }
    )
  
    dataset=dataset.map(input_prompt)


    for split in ['train','validation','challenge']:
      output_dir = f'features/{model_name.split("/")[1]}'
      os.makedirs(output_dir, exist_ok=True)

      layer_features = get_hidden_states(dataset[split]['messages'],model,tokenizer,device)
      torch.save(layer_features, f'{output_dir}/{split}.features')

if __name__ == "__main__":
    main()
