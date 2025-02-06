import argparse
import os
import pickle
import warnings
from time import time

import warnings
import pandas as pd
import torch
from omegaconf import OmegaConf
from dataloader.dataloader import DTIDataset, get_dataLoader
from models.transformer_dti import TransformerDTI
from trainer import Trainer
from utils.utils import set_seed, mkdir, load_config_file
from transformers import AutoModel, AutoTokenizer
from models.kmeans_for_confoudners import kmeans_confounder

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['random', 'cold', 'cluster', 'augmented'])
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_CONFIG_PATH = 'configs/train_config.yaml'
MODEL_CONFIG_PATH = 'configs/model_config.yaml'
print(f"Running on: {device}", end="\n\n")

def main():
    torch.cuda.empty_cache()
    warnings.simplefilter("ignore")

    train_config = load_config_file(TRAIN_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)
    config = OmegaConf.merge(train_config, model_config)
    model_configs = dict(model_config)
    set_seed(seed=config.TRAIN.SEED)
    output_path = f"./results/{args.data}/{args.split}/{config.TRAIN.OUTPUT_DIR}"
    mkdir(output_path)

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))
    stage = 2

    if args.split == 'cluster':
        train_path = os.path.join(dataFolder, 'source_train_with_id.csv')
        val_path = os.path.join(dataFolder, "target_test_with_id.csv")
        test_path = os.path.join(dataFolder, "target_test_with_id.csv")
    else:
        train_path = os.path.join(dataFolder, 'train_with_id.csv')
        val_path = os.path.join(dataFolder, "val_with_id.csv")
        test_path = os.path.join(dataFolder, "test_with_id.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    protein_path = os.path.join(dataFolder, config.TRAIN.PR_PATH)
    protein_f = open(protein_path, 'rb')
    pr_f = pickle.load(protein_f)

    mol_path = 'models/drug/molformer'
    drug_tokenizer = AutoTokenizer.from_pretrained(mol_path, trust_remote_code=True)

    if stage == 1:
        model = TransformerDTI(model_configs=model_configs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        checkpoint_path = f'./results/{args.data}/random/0.7_random_8_123/stage_1_best_epoch_92.pth'
        kmeans_dataset = DTIDataset(df_train.index.values, df_train, pr_f)
        kmeans_dataloader = get_dataLoader(1, kmeans_dataset, drug_tokenizer)
        kmeans_confounder(device, stage, model_config, train_config, kmeans_dataloader, output_path + '/',
                          checkpoint_path)

        pr_confounder_path = os.path.join(output_path, config.TRAIN.PR_CONFOUNDER_PATH)
        confounder_path = open(pr_confounder_path, 'rb')
        confounder = pickle.load(confounder_path)
        pr_confounder = torch.from_numpy(confounder['cluster_centers']).to(device)
        # pr_confounder = confounder['cluster_centers'].to(device)

        model = TransformerDTI(pr_confounder=pr_confounder,
                               model_configs=model_configs).to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY*5)





    train_dataset = DTIDataset(df_train.index.values, df_train, pr_f)
    val_dataset = DTIDataset(df_val.index.values, df_val, pr_f)
    test_dataset = DTIDataset(df_test.index.values, df_test, pr_f)


    bz = config.TRAIN.BATCH_SIZE
    train_dataloader = get_dataLoader(bz, train_dataset, drug_tokenizer, shuffle=True)
    val_dataloader = get_dataLoader(bz, val_dataset, drug_tokenizer)
    test_dataloader = get_dataLoader(bz, test_dataset, drug_tokenizer)

    trainer = Trainer(model, opt, device, stage, train_dataloader, val_dataloader, test_dataloader, output_path, config)
    result, best_epoch = trainer.train()

    with open(os.path.join(output_path, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    print()
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
