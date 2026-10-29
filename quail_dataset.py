from datasets import load_dataset

ds = load_dataset("textmachinelab/quail")

def data_map(dataset):
  dataset['skill_type'] = ['Abstain' if q_type == 'Unanswerable' else 'Refer' if q_type in ['Factual','Temporal_order','Character_identity'] else 'Infer' for q_type in dataset['question_type']]
  return dataset

ds=ds.map(data_map, batched= True)

ds['train'].to_json("dataset/train.json")
ds['validation'].to_json("dataset/validation.json")
ds['challenge'].to_json("dataset/challenge.json")