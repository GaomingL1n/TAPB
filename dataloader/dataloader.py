import torch
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import Dataset, DataLoader
import random
from rdkit import Chem
import numpy as np
class DTIDataset(Dataset):
    def __init__(self, list_IDs, df, pr_features):
        self.list_IDs = list_IDs
        self.df = df
        #self.drug_ids = drug_ids
        self.pr_features = pr_features

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        index = self.list_IDs[idx]
        # smiles
        SMILES = self.df.iloc[index]['SMILES']
        # SMILES2 = randomize_smile(SMILES)


        # proteins
        pr_id = self.df.iloc[index]['pr_id']
        Protein = self.pr_features[pr_id]
        # labels
        y = self.df.iloc[index]["Y"]

        return {
            'SMILES': SMILES,
            # 'SMILES2': SMILES2,
            'Protein': Protein,
            'Y': y
        }

def randomize_smile(sml):
    """Function that randomizes a SMILES sequence. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequence to randomize.
    Return:
        randomized SMILES sequence or
        nan if SMILES is not interpretable.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        smiles = Chem.MolToSmiles(nm, canonical=False)

        return smiles

    except:
        return sml
def convert_batch_pr(batch):
    max_len = max([tensor.shape[0] for tensor in batch])
    mask = torch.zeros(len(batch), max_len)
    for i, tensor in enumerate(batch):
        mask[i, :tensor.shape[0]] = 1
    padded_batch = pad(batch, batch_first=True)
    return {'input_ids': padded_batch, 'attention_mask': mask}
def mask_tokens(inputs, attention_mask, tokenizer, probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    mask = attention_mask.clone()
    mask[:, 0] = 0  # cls token
    eos_pos = mask.sum(dim=1).unsqueeze(1)
    mask.scatter_(1, eos_pos, 0)
    mask = mask.bool()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, probability)).bool()
    masked_indices = masked_indices * mask
    labels[~masked_indices] = -1  # We only compute loss on masked tokens
    #pos = (labels + 1).nonzero()[;]
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def drop_tokens(batch, drop_prob=0.75):
    batch_id, batch_mask = batch['input_ids'], batch['attention_mask']
    num_to_retain = int(max(1, batch_id.size(1) * (1-drop_prob))) # 确保至少保留1个词

    # 随机选择要保留的索引，确保每个索引只被选中一次
    indices = sorted(random.sample(range(1, batch_id.size(1)), num_to_retain))
    indices.insert(0, 0)
    # 根据选中的索引保留相应的词
    batch_id = batch_id[:, indices]
    batch_mask = batch_mask[:, indices]
    # 将结果转换回Python列表并返回
    return batch_id, batch_mask

def get_dataLoader(config, stage, dataset, drug_tokenizer, shuffle=False):
    def collate_fn(batch_samples):
        batch_Drug, batch_Drug2, batch_Protein, batch_label = [], [], [], []
        for sample in batch_samples:
            batch_Drug.append(sample['SMILES'])
            # batch_Drug2.append(sample['SMILES2'])
            batch_Protein.append(sample['Protein'])
            batch_label.append(sample['Y'])
        batch_inputs_drug = drug_tokenizer(batch_Drug, padding='longest', return_tensors="pt")
        # batch_inputs_drug2 = drug_tokenizer(batch_Drug2, padding='longest', return_tensors="pt")
        batch_pr = convert_batch_pr(batch_Protein)
        if shuffle & stage != 3:
            #batch_pr['input_ids'], batch_pr['attention_mask'] = drop_tokens(batch_pr, 0.5)
            batch_inputs_drug['input_ids'], batch_inputs_drug['attention_mask'] = drop_tokens(batch_inputs_drug, 0.1)
        if stage == 1:
            batch_inputs_drug['masked_input_ids'], batch_inputs_drug['masked_drug_labels']\
                = mask_tokens(batch_inputs_drug['input_ids'], batch_inputs_drug['attention_mask'], drug_tokenizer)
        return {
            'batch_inputs_drug': batch_inputs_drug,
            # 'batch_inputs_drug2':batch_inputs_drug2,
            'batch_inputs_pr': batch_pr,
            'labels': batch_label
        }

    # return DataLoader(dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=shuffle, collate_fn=collate_fn,
    #                  num_workers=config.TRAIN.NUM_WORKERS, pin_memory=True, persistent_workers=True)
    return DataLoader(dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=shuffle, collate_fn=collate_fn)