# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch.nn.functional as F
    

class DecoderClassifier(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DecoderClassifier, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.classifier = nn.Linear(config.hidden_size, 2)
        
    def forward(self, input_ids=None, labels=None, weight=None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        logits = self.classifier(hidden_states)

        batch_size = input_ids.size(0)
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        prob = nn.functional.softmax(pooled_logits, dim=-1)


        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)

            loss = loss_fct(pooled_logits.view(-1, 2), labels.view(-1))
            return loss, prob
        else:
            return prob