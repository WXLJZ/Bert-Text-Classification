# coding=utf-8

from transformers import BertModel, BertPreTrainedModel

from torch import nn
from torch.nn import CrossEntropyLoss

class BertOrigin(BertPreTrainedModel):

    def __init__(self, config):
        super(BertOrigin, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """ 
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        # pooled_output: [batch_size, dim=768]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        # logits: [batch_size, output_dim=2]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
