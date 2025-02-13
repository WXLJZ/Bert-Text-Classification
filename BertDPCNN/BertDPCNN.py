# coding=utf-8

from transformers import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class BertDPCNN(BertPreTrainedModel):

    def __init__(self, config, filter_num):
        super(BertDPCNN, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.conv_region = nn.Conv2d(1, filter_num, (3, config.hidden_size), stride=1)
        self.convs = nn.Conv2d(filter_num, filter_num, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3,1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fn = nn.ReLU()
        self.classifier = nn.Linear(filter_num, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """ 
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        sequence_output = self.dropout(outputs[0])

        sequence_output = sequence_output.unsqueeze(1)
        x = self.conv_region(sequence_output)
        # x: [batch_size, filter_num, seq_len-3+1, bert_dim-bert_dim+1=1]

        x = self.padding_conv(x)
        # x: [batch_size, filter_num, seq_len, 1]

        x = self.act_fn(x)

        x = self.convs(x)
        # x : [batch_size, filter_num, seq_len-3+1, 1-1+1=1]

        x = self.padding_conv(x)
        # x: [batch_size, filter_num, seq_len, 1]

        x = self.act_fn(x)

        x = self.convs(x)
        # x : [batch_size, filter_num, seq_len-3+1, 1-1+1=1]

        while x.size()[-2] > 2:
            x = self._block(x)
        
        x = x.squeeze()
        # x: [batch_size, filter_num]

        logits = self.classifier(x)
        # logits: [batch_size, output_dim]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
    
    def _block(self, x):
        
        # x: [batch_size, filter_num, len, 1]
        x = self.padding_pool(x)
        # x: [batch_size, filter_num, len+1, 1]

        px = self.pooling(x)
        # px: [batch_size, filter_num, ((len+1)-3)//2 + 1, 1-1+1=1]

        x = self.padding_conv(px)
        # x: [batch_size, filter_num, ((len+1)-3)//2+1+2 = ((len+1)-3)//2+3, 1]
        x = F.relu(x)

        x = self.convs(x)
        # x: [batch_size, filter_num, ((len+1)-3)//2+3-3+1=((len+1)-3)//2+1, 1-1+1=1]

        x = self.padding_conv(x)
        # x: [batch_size, filter_num, ((len+1)-3)//2+1+2=((len+1)-3)//2+3, 1]
        x = F.relu(x)
        x = self.convs(x)
        # x: [batch_size, filter_num, ((len+1)-3)//2+3-3+1=((len+1)-3)//2+1, 1]

        x = x + px
        # x: [batch_size, filter_num, ((len+1)-3)//2+1, 1]
        return x

