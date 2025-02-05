import argparse
import os
import pickle
import warnings
from time import time

import pandas as pd
import torch
from omegaconf import OmegaConf
from dataloader.dataloader import DTIDataset, get_dataLoader
from models.drug_encoder.tokenizer.tokenizer import MolTranBertTokenizer
from models.transformer_dti import TransformerDTI
from trainer import Trainer
from utils.utils import set_seed, mkdir, load_config_file

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
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    stage = 1
    train_config = load_config_file(TRAIN_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)
    config = OmegaConf.merge(train_config, model_config)
    model_configs = dict(model_config)
    set_seed(seed=2048)
    # suffix = str(int(time() * 1000))[6:]
    mkdir(config.TRAIN.OUTPUT_DIR)

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))
    checkpoint_path = f'./results/{args.data}/{args.split}/plain/stage_1_best_epoch_294.pth'
    pr_confounder_path = os.path.join(dataFolder, config.TRAIN.PR_CONFOUNDER_PATH)
    confounder_path = open(pr_confounder_path, 'rb')
    pr_confounder = torch.from_numpy(pickle.load(confounder_path)['cluster_centers']).to(device)
    if stage == 1:
        model = TransformerDTI(
            # pr_confounder=pr_confounder,
                                model_configs=model_configs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        confounder_path = os.path.join(dataFolder, config.TRAIN.DRUG_CONFOUNDER_PATH)
        drug_confounder = open(confounder_path, 'rb')
        drug_confounder = torch.from_numpy(pickle.load(drug_confounder)['cluster_centers']).to(device)
        if stage == 2:
            model = TransformerDTI(pr_confounder=pr_confounder,
                                   drug_confounder=drug_confounder,
                                   model_configs=model_configs).to(device)
        elif stage == 3:
            confounder_path = os.path.join(dataFolder, config.TRAIN.FUSION_CONFOUNDER_PATH)
            fusion_confounder = open(confounder_path, 'rb')
            fuison_confounder = torch.from_numpy(pickle.load(fusion_confounder)['cluster_centers']).to(device)

            model = TransformerDTI(pr_confounder=pr_confounder,
                                   #drug_confounder=drug_confounder,
                                   fusion_confounder=fuison_confounder,
                                   model_configs=model_configs).to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    train_path = os.path.join(dataFolder, 'train_with_id.csv')
    val_path = os.path.join(dataFolder, "val_with_id.csv")
    test_path = os.path.join(dataFolder, "test_with_id.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    protein_path = os.path.join(dataFolder, config.TRAIN.PR_PATH)
    protein_f = open(protein_path, 'rb')
    pr_f = pickle.load(protein_f)

    train_dataset = DTIDataset(df_train.index.values, df_train, pr_f)
    val_dataset = DTIDataset(df_val.index.values, df_val, pr_f)
    test_dataset = DTIDataset(df_test.index.values, df_test, pr_f)

    drug_tokenizer = MolTranBertTokenizer()

    train_dataloader = get_dataLoader(config, stage, train_dataset, drug_tokenizer, shuffle=True)
    val_dataloader = get_dataLoader(config, stage, val_dataset, drug_tokenizer, shuffle=False)
    test_dataloader = get_dataLoader(config, stage, test_dataset, drug_tokenizer, shuffle=False)

    trainer = Trainer(model, opt, device, stage, train_dataloader, val_dataloader, test_dataloader, config)
    result, best_epoch = trainer.train()

    with open(os.path.join(config.TRAIN.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    print()
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
