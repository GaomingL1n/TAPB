import torch
import numpy as np
import os
import pickle
import pandas as pd
from tqdm import tqdm
from time import time
from sklearn.cluster import KMeans
from dataloader.dataloader import DTIDataset, get_dataLoader
from models.transformer_dti import TransformerDTI
from transformers import AutoTokenizer
from utils.utils import load_config_file

def kmeans_stage2(featurelist, config, d_model):
    matrix = np.vstack(featurelist)

    kmeans = KMeans(n_clusters=config.TRAIN.DICT_SIZE)
    kmeans.fit(matrix)

    labels = kmeans.labels_

    cluster_centers = np.zeros((d_model, config.TRAIN.DICT_SIZE))

    prior = list()
    total = 0
    for i in range(config.TRAIN.DICT_SIZE):
        cluster_position = matrix[labels == i]
        num = cluster_position.shape[0]
        prior.append(num)
        total += num
        cluster_centers[:, i] = np.mean(cluster_position, axis=0)
    prior = torch.tensor(prior)
    prior = prior / total
    cluster_dict = dict()
    cluster_dict['cluster_centers'] = cluster_centers
    cluster_dict['prior'] = prior
    return cluster_dict

def kmeans_confounder(device, stage, model_configs, train_config, train_dataloader, save_path, best_model_path):
    device = device
    torch.cuda.empty_cache()
    d_model = model_configs['DrugEncoder']['d_model']
    model = TransformerDTI(model_configs=model_configs).to(device)
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    featurelist = list()
    featurelist_neg = list()
    featurelist2 = list()
    featurelist2_neg = list()
    pr_pos = torch.zeros(1,128)
    pr_neg = torch.zeros(1,128)
    drug_pos = torch.zeros(1,128)
    drug_neg = torch.zeros(1,128)
    pos_count = 0
    neg_count = 0
    for batch in tqdm(train_dataloader):
        input_drugs = batch['batch_inputs_drug'].to(device)
        input_proteins = batch['batch_inputs_pr']['input_ids'].to(device)
        pr_mask = batch['batch_inputs_pr']['attention_mask'].to(device)
        y = batch['labels'][0]
        with torch.no_grad():
            # if stage ==1:
            fusion_f = model(input_drugs,input_proteins,pr_mask = pr_mask)['fusion_f']
            # else:
            # drug_f = model.encode_drug(input_drugs, model.precompute_freqs_cis.to(device))
            # pr_f_fake = torch.ones_like(pr_f)
            # fusion_f,_ = model.decode(drug_f, pr_f, input_drugs['attention_mask'], pr_mask)
            #
            featurelist.append(fusion_f.cpu())
            # drug_f_fake = torch.ones_like(drug_f)
            # fusion_f,_ = model.decode(drug_f_fake, pr_f, input_drugs['attention_mask'], pr_mask)
            # featurelist2.append(pr_f.mean(1).cpu())


            # if y == 1:
            #     # drug_pos += drug_f.mean(1).cpu()
            #     pr_pos += fusion_f.mean(1).cpu()
            #     pos_count += 1
            #     # featurelist.append(pr_f.mean(1).cpu())
            #     # featurelist2.append(drug_f.mean(1).cpu())
            # else:
            #     # drug_neg += drug_f.mean(1).cpu()
            #     pr_neg += fusion_f.mean(1).cpu()
            #     neg_count += 1
                # featurelist_neg.append(pr_f.mean(1).cpu())
                # featurelist2_neg.append(drug_f.mean(1).cpu())
    # pr_pos = pr_pos/pos_count
    # pr_neg = pr_neg/neg_count
    # drug_pos = drug_pos / pos_count
    # drug_neg = drug_neg / neg_count
    # if stage==1:
    save_path = save_path + train_config.TRAIN.PR_CONFOUNDER_PATH
    # else:
    # save_path2 = path + train_config.TRAIN.DRUG_CONFOUNDER_PATH

    cluster_dict = kmeans_stage2(featurelist, train_config, d_model)
    # cluster_centers = cluster_dict['cluster_centers']
    # cluster_dict = kmeans_for_stage2(featurelist_neg, train_config)
    # cluster_centers2 = cluster_dict['cluster_centers']
    # cluster_dict['cluster_centers'] = np.hstack((cluster_centers, cluster_centers2))
    # cluster_dict = dict()
    # cluster_dict['cluster_centers'] = torch.cat((pr_neg,pr_pos),0)
    file = open(save_path, 'wb')
    pickle.dump(cluster_dict, file)

    # cluster_dict = kmeans_stage2(featurelist2, train_config, d_model)
    # # # cluster_centers = cluster_dict['cluster_centers']
    # # cluster_dict = kmeans_for_stage2(featurelist2_neg, train_config)
    # # cluster_centers2 = cluster_dict['cluster_centers']
    # # cluster_dict['cluster_centers'] = np.hstack((cluster_centers, cluster_centers2))
    # # cluster_dict['cluster_centers'] = torch.cat((drug_pos, drug_neg), 0)
    # file = open(save_path2, 'wb')
    # pickle.dump(cluster_dict, file)


# generate confounders after stage 1 training
if __name__ == '__main__':
    s = time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TRAIN_CONFIG_PATH = '../configs/train_config.yaml'
    MODEL_CONFIG_PATH = '../configs/model_config.yaml'
    train_config = load_config_file(TRAIN_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)
    datasets = 'bindingdb'
    split = 'random'
    res = 'ex2048_6'
    checkpoint_path = f"../results/{datasets}/{split}/{res}/stage_1_best_epoch_99.pth"
    dataFolder = f'../datasets/{datasets}/{split}'

    protein_path = os.path.join(dataFolder, train_config.TRAIN.PR_PATH)
    protein_f = open(protein_path, 'rb')
    pr_f = pickle.load(protein_f)
    if split == 'random':
        train_path = os.path.join(dataFolder, 'train_with_id.csv')
    else:
        train_path = os.path.join(dataFolder, 'source_train_with_id.csv')
    df_train = pd.read_csv(train_path)
    train_dataset = DTIDataset(df_train.index.values, df_train, pr_f)
    mol_path = '../models/drug/molformer'
    drug_tokenizer = AutoTokenizer.from_pretrained(mol_path, trust_remote_code=True)
    train_dataloader = get_dataLoader(1, train_dataset, drug_tokenizer)
    pr_dataFolder = f"../results/{datasets}/{split}/{res}/"
    kmeans_confounder(device, 1, model_config, train_config, train_dataloader, pr_dataFolder, checkpoint_path)
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")