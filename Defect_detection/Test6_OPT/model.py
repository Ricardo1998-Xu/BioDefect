# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Focal loss + Class Weights
class FocalLoss_Weight(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss_Weight, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_weight = self.alpha[target]
            focal_loss *= alpha_weight

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

        
    def forward(self, input_ids=None,labels=None): 
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1), output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = self.dropout(hidden_states)

        last_token_logits = hidden_states[:, -1, :]  # (batch_size, hidden_size)

        logits = self.classifier(last_token_logits)  # (batch_size, 2)

        prob = nn.functional.softmax(logits, dim=-1)
        if labels is not None:

            ## Class Weights
            # class_weights = torch.tensor([1.0, 10.0]).to("cuda")
            # loss_fct = nn.CrossEntropyLoss(weight=class_weights)

            ## Focal loss
            # loss_fct = FocalLoss()

            ## Focal loss + Class Weights
            # loss_fct = FocalLoss_Weight(alpha=class_weights)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob