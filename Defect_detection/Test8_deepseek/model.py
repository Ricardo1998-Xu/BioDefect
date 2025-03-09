# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
import torch.utils.checkpoint as checkpoint
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


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

    def custom_forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states[-1]

    def forward(self, input_ids=None, labels=None):
        attention_mask = input_ids.ne(1)

        # use checkpoint
        hidden_states = checkpoint.checkpoint(self.custom_forward, input_ids, attention_mask)
        hidden_states = self.dropout(hidden_states)
        last_token_logits = hidden_states[:, -1, :]  # (batch_size, hidden_size)

        logits = self.classifier(last_token_logits)  # (batch_size, 2)

        prob = nn.functional.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            # loss_fct = FocalLoss()
            # loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob