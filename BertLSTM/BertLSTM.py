from transformers import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from Models.Linear import Linear

class BertLSTM(BertPreTrainedModel):

    def __init__(self, config, rnn_hidden_size, num_layers, bidirectional, dropout):
        super(BertLSTM, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers,bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(rnn_hidden_size * 2, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False,
            output_hidden_states=True
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # encoded_layers: [batch_size, seq_len, bert_dim]

        _, (hidden, cell) = self.rnn(sequence_output)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        logits = self.classifier(hidden)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
