import sys

import torch
import torch.nn as nn
from torch import einsum
import copy
import os
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from prettytable import PrettyTable
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, optim, device, stage, train_dataloader, val_dataloader, test_dataloader, output_path, config):
        self.model = model
        self.optim = optim
        self.device = device
        self.batch_size = config.TRAIN.BATCH_SIZE
        if stage == 1:
            self.epochs = config.TRAIN.Stage1_MAX_EPOCH
        else:
            self.epochs = config.TRAIN.Stage2_MAX_EPOCH

        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.step = 0

        self.nb_training = len(self.train_dataloader)

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = output_path
        self.stage = stage
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold"]

        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc = self.test(dataloader="val")

            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc]))
            self.val_table.add_row(val_lst)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print('Stage ' + str(self.stage) + ' Validation at Epoch ' + str(self.current_epoch), "with AUROC "+ str(auroc) + " AUPRC " + str(auprc))
        auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + " with AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()

        return self.test_metrics, self.best_epoch

    def save_result(self):
        if self.config.TRAIN.SAVE_MODEL:
            torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, f"stage_{self.stage}_best_epoch_{self.best_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))
        if self.config.TRAIN.SAVE_LAST_EPOCH:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"stage_{self.stage}_last_epcoh.pth"))

        val_prettytable_file = os.path.join(self.output_dir, 'Stage_' + str(self.stage) +"_valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, 'Stage_' + str(self.stage) +"_test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, 'Stage_' + str(self.stage) +"_train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        r=0
        num_batches = len(self.train_dataloader)
        loop = tqdm(self.train_dataloader, colour='#ff4777', file=sys.stdout, ncols=150)
        loop.set_description(f'Stage {self.stage} Train Epoch[{self.current_epoch}/{self.epochs}]')
        for step, batch in enumerate(loop):
            self.optim.zero_grad()
            self.step += 1
            r+=1
            input_drugs = batch['batch_inputs_drug'].to(self.device)
            input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
            pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
            labels = torch.tensor(batch['labels']).to(self.device)

            if self.stage == 1:
                inputs_drugs_m = batch['batch_inputs_drug_m'].to(self.device)
                drug_labels = batch['masked_drug_labels'].to(self.device)
                output = self.model(input_drugs, input_proteins, pr_mask=pr_mask, masked_drugs=inputs_drugs_m)
                mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)(output['drug_mlm_logits'], drug_labels)
                # labels = torch.cat((labels,torch.zeros_like(labels)),0)
                b_loss = cross_entropy(output['logits'], labels)
                # b_loss2 = cross_entropy(output['pr_logits'], labels)
                # b_loss3 = cross_entropy(output['drug_logits'], labels)
                # loss = b_loss
                loss = b_loss+ mlm_loss
                # loss = b_loss + mlm_loss + b_loss2 +b_loss3
            else:
                output = self.model(input_drugs, input_proteins, pr_mask=pr_mask)
                # b_label = torch.cat((labels, torch.zeros_like(labels)), 0)
                # loss = cross_entropy(output['logits'], b_label)
                loss = cross_entropy(output['logits'], labels)

            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            loop.set_postfix(avg_loss=loss_epoch / r)

        loss_epoch = loss_epoch / num_batches
        return loss_epoch

    def test(self, dataloader="test"):
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        loop = tqdm(data_loader, colour='#f47983', file=sys.stdout)
        with torch.no_grad():
            self.model.eval()
            if dataloader == "val":
                loop.set_description(f'Stage {self.stage} Validation')
            elif dataloader == "test":
                loop.set_description(f'Stage {self.stage} Test')
            for step, batch in enumerate(loop):
                input_drugs = batch['batch_inputs_drug'].to(self.device)
                input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
                pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
                labels = batch['labels']

                if dataloader == "val":
                    output = self.model(input_drugs, input_proteins, pr_mask=pr_mask)

                elif dataloader == "test":
                    output = self.best_model(input_drugs, input_proteins, pr_mask=pr_mask)

                n = output['logits'][:, 1]
                y_label = y_label + labels
                y_pred = y_pred + n.tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.000001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, thred_optim, precision1
        else:
            return auroc, auprc

def binary_cross_entropy(n, labels):
    loss_fct = torch.nn.BCELoss()
    loss = loss_fct(n, labels.float())
    return loss

def cross_entropy(preds, targets, reduction='none'):
    preds = torch.log(preds)
    loss_f = nn.NLLLoss()
    loss = loss_f(preds, targets.long())
    return loss