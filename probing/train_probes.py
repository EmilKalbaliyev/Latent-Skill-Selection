from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle
import pandas as pd 
from collections import Counter
import random
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import os


def process_layer(
    layer_idx,
    X_train_layer,
    y_train,
    X_validation_layer,
    y_validation,
    X_challenge_layer,
    y_challenge,
    prob_type,
    model_name,
    s_type,
):
    if prob_type != "sklearn_linear_model":
        return

    clf = LogisticRegression(
        max_iter=10000,
        penalty="l2",
        n_jobs=1,
    )

    clf.fit(X_train_layer, y_train)

    with open(
        f"probs/{prob_type}/{model_name}/{s_type}/layer_{layer_idx}.pkl",
        "wb",
    ) as f:
        pickle.dump(clf, f)

    df = pd.DataFrame(
        {
            "label": y_validation,
            "pred": clf.predict(X_validation_layer),
        }
    )

    df.to_csv(
        f"preds/{prob_type}/{model_name}/{s_type}/validation/layer_{layer_idx}.csv",
        index=False,
    )

    df = pd.DataFrame(
        {
            "label": y_challenge,
            "pred": clf.predict(X_challenge_layer),
        }
    )

    df.to_csv(
        f"preds/{prob_type}/{model_name}/{s_type}/challenge/layer_{layer_idx}.csv",
        index=False,
    )

    return layer_idx

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Language model from Hugging Face"
    )

    parser.add_argument(
        "--prob_type",
        type=str,
        required=True,
    )
    set_seed(42)
    args = parser.parse_args()
    model_name = args.model_name
    prob_type = args.prob_type
    output_dir = f'features/{model_name}'#CHANGE

    for s_type in ['skill','gap','sufficiency','Refer-Infer','Refer-Abstain', 'Infer-Abstain']:
        data = load_dataset(
            'json',
            data_files={
                'train': '../dataset/train.json',
                'validation': '../dataset/validation.json',
                'challenge': '../dataset/challenge.json',
            })

        X_features = { 
        'train': torch.load(f'{output_dir}/train.features'),
        'validation': torch.load(f'{output_dir}/validation.features'),
        'challenge' : torch.load(f'{output_dir}/challenge.features')}


        if s_type=='skill':
            SKILL2ID = { "Refer": 0, "Infer": 1, "Abstain": 2 }
            
            Y_labels = {'train' : [SKILL2ID[i] for i in data['train']['skill_type']],
                        'validation' : [SKILL2ID[i] for i in data['validation']['skill_type']],
                        'challenge' : [SKILL2ID[i] for i in data['challenge']['skill_type']]
            }
            num_classes=3
        
        elif s_type=='gap':
            SKILL2ID = { "Refer": 0, "Infer": 1, "Abstain": 1 }
            
            Y_labels = {'train' : [SKILL2ID[i] for i in data['train']['skill_type']],
                        'validation' : [SKILL2ID[i] for i in data['validation']['skill_type']],
                        'challenge' : [SKILL2ID[i] for i in data['challenge']['skill_type']]
            }
            num_classes=2

        elif s_type=='sufficiency':

            SKILL2ID = { "Refer": 0, "Infer": 0, "Abstain": 1 }
            
            Y_labels = {'train' : [SKILL2ID[i] for i in data['train']['skill_type']],
                        'validation' : [SKILL2ID[i] for i in data['validation']['skill_type']],
                        'challenge' : [SKILL2ID[i] for i in data['challenge']['skill_type']]
            }
            num_classes=2
        
        elif '-' in s_type:
            sub_skills=s_type.split('-')
            SKILL2ID = { sub_skills[0]: 0, sub_skills[1]: 1 }

            for split in data.keys():
                mask = [x in sub_skills for x in data[split]["skill_type"]]
                data[split] = data[split].select([i for i, m in enumerate(mask) if m])
                X_features[split] = X_features[split][mask]

            Y_labels = {'train' : [SKILL2ID[i] for i in data['train']['skill_type']],
                        'validation' : [SKILL2ID[i] for i in data['validation']['skill_type']],
                        'challenge' : [SKILL2ID[i] for i in data['challenge']['skill_type']]
            }
            num_classes=2


        os.makedirs(f'probs/{prob_type}/{model_name}/{s_type}', exist_ok=True)
        os.makedirs(f'preds/{prob_type}/{model_name}/{s_type}/validation', exist_ok=True)
        os.makedirs(f'preds/{prob_type}/{model_name}/{s_type}/challenge', exist_ok=True)

        num_layers = X_features['train'].shape[1]
        print(num_layers)

        with tqdm_joblib(tqdm(desc=s_type, total=num_layers)):
            Parallel(n_jobs=-1, backend="loky")(
                delayed(process_layer)(
                    layer_idx,
                    X_features["train"][:, layer_idx, :].numpy(),
                    Y_labels["train"],
                    X_features["validation"][:, layer_idx, :].numpy(),
                    Y_labels["validation"],
                    X_features["challenge"][:, layer_idx, :].numpy(),
                    Y_labels["challenge"],
                    prob_type,
                    model_name,
                    s_type,
                )
                for layer_idx in range(num_layers)
            )

if __name__ == "__main__":
    main()
