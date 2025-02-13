# coding=utf-8
import random
import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import BertTokenizer, BertConfig


from Utils.utils import get_device
from Utils.load_datatsets import load_data
from Utils.lr_scheduler import get_linear_schedule_with_warmup
from Utils.logger import get_logger

from train_evalute import train, evaluate, evaluate_save

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)

def main(args, model_times, label_list):

    if not os.path.exists(args.output_dir + model_times):
        os.makedirs(args.output_dir + model_times)

    if not os.path.exists(args.cache_dir + model_times):
        os.makedirs(args.cache_dir + model_times)

    # Bert 模型输出文件
    output_dir = os.path.join(args.output_dir, model_times)

    # 设备准备
    gpu_ids = [int(device_id) for device_id in args.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])  
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # 设定随机种子 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"----> Loading config & tokenizer from {args.model_name_or_path}.....")
    logger.info("----> Loading Config.....")
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(label_list))
    logger.info("----> Loading tokenizer.....")
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case)  # 分词器选择
    logger.info("----> Tokenizer loaded!")

    # Train and dev
    if args.do_train:
        logger.info("----> Loading Train and Dev data.....")
        train_dataloader, train_examples_len = load_data(
            args.data_dir, tokenizer, args.max_seq_length, args.train_batch_size, "train", label_list)
        dev_dataloader, dev_examples_len = load_data(
            args.data_dir, tokenizer, args.max_seq_length, args.dev_batch_size, "dev", label_list)
        logger.info(f"----> Train and Dev data loaded! Train data size is {train_examples_len}, Dev data size is {dev_examples_len}")
        
        num_train_optimization_steps = int(
            train_examples_len / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        
        # 模型准备
        logger.info("model name is {}".format(args.model_name))
        logger.info("----> Loading Model.....")
        if args.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin.from_pretrained(
                args.model_name_or_path,
                config=config
            )
        elif args.model_name == "BertCNN":
            from BertCNN.BertCNN import BertCNN
            filter_sizes = [int(val) for val in args.filter_sizes.split()]
            model = BertCNN.from_pretrained(
                args.model_name_or_path,
                config=config,
                n_filters=args.filter_num,
                filter_sizes=filter_sizes
            )
        elif args.model_name == 'BertLSTM':
            from BertLSTM.BertLSTM import BertLSTM
            model = BertLSTM.from_pretrained(
                args.model_name_or_path,
                config=config,
                rnn_hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                bidirectional=args.bidirectional,
                dropout=args.dropout
            )

        elif args.model_name == "BertATT":
            from BertATT.BertATT import BertATT
            model = BertATT.from_pretrained(
                args.model_name_or_path,
                config=config
            )
        
        elif args.model_name == "BertRCNN":
            from BertRCNN.BertRCNN import BertRCNN
            model = BertRCNN.from_pretrained(
                args.model_name_or_path,
                config=config,
                rnn_hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                bidirectional=args.bidirectional,
                dropout=args.dropout
            )

        elif args.model_name == "BertCNNPlus":
            from BertCNNPlus.BertCNNPlus import BertCNNPlus
            filter_sizes = [int(val) for val in args.filter_sizes.split()]
            model = BertCNNPlus.from_pretrained(
                args.model_name_or_path,
                config=config,
                n_filters=args.filter_num,
                filter_sizes=filter_sizes
            )
        elif args.model_name == "BertDPCNN":
            from BertDPCNN.BertDPCNN import BertDPCNN
            model = BertDPCNN.from_pretrained(
                args.model_name_or_path,
                config=config,
                filter_num=args.filter_num
            )

        model.to(device)
        logger.info("----> Model loaded!")

        if n_gpu > 1:
            model = torch.nn.DataParallel(model,device_ids=gpu_ids)
        """ 优化器准备 """
        bert_param_optimizer = list(model.bert.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 1e-5},
            {'params': [p for n, p in linear_param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 1e-5}
        ]
        if args.model_name == 'BertLSTM' or args.model_name == 'BertRCNN':
            optimizer_grouped_parameters.extend([
                {'params': [p for n, p in list(model.rnn.named_parameters()) if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 1e-5},
                {'params': [p for n, p in list(model.rnn.named_parameters()) if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 1e-5}
            ])
        elif args.model_name == 'BertCNN' or args.model_name == 'BertDPCNN' or args.model_name == 'BertCNNPlus':
            optimizer_grouped_parameters.extend([
                {'params': [p for n, p in list(model.convs.named_parameters()) if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 1e-5},
                {'params': [p for n, p in list(model.convs.named_parameters()) if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 1e-5}
            ])
        args.warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        """ 损失函数准备 """
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        logger.info("----> Start training.....")
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Total Parameters = {:,}".format(sum(p.numel() for p in model.parameters())))
        logger.info("  Num examples = %d", train_examples_len)
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_train_optimization_steps)
        train(args=args,
              epoch_num=args.num_train_epochs,
              n_gpu=n_gpu,
              model=model,
              tokenizer=tokenizer,
              train_dataloader=train_dataloader,
              dev_dataloader=dev_dataloader,
              optimizer=optimizer,
              scheduler=scheduler,
              criterion=criterion,
              gradient_accumulation_steps=args.gradient_accumulation_steps,
              device=device,
              label_list=label_list,
              output_dir=output_dir,
              log_dir=args.log_dir,
              print_step=args.print_step,
              early_stop=args.early_stop)
        logger.info("----> Training finished!")

    elif not args.do_train:
        # Test
        logger.info("----> Start testing.....")
        logger.info("----> Loading Test data.....")

        logger.info(f"----> Loading Config and Tokenizer from {output_dir}.....")
        config = BertConfig.from_pretrained(output_dir, num_labels=len(label_list))
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        logger.info("----> BertConfig and Tokenizer loaded!")

        test_dataloader, test_examples_len = load_data(args.data_dir, tokenizer, args.max_seq_length, args.test_batch_size, "test", label_list)
        logger.info(f"----> Test data loaded! Test data size is {test_examples_len}")

        logger.info(f"----> Loading Model from {output_dir}.....")
        if args.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin.from_pretrained(
                output_dir,
                config=config
            )
        elif args.model_name == "BertCNN":
            from BertCNN.BertCNN import BertCNN
            filter_sizes = [int(val) for val in args.filter_sizes.split()]
            model = BertCNN.from_pretrained(
                output_dir,
                config=config,
                n_filters=args.filter_num,
                filter_sizes=filter_sizes
            )
        elif args.model_name == 'BertLSTM':
            from BertLSTM.BertLSTM import BertLSTM
            model = BertLSTM.from_pretrained(
                output_dir,
                config=config,
                rnn_hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                bidirectional=args.bidirectional,
                dropout=args.dropout
            )
        elif args.model_name == "BertATT":
            from BertATT.BertATT import BertATT
            model = BertATT.from_pretrained(
                output_dir,
                config=config
            )
        elif args.model_name == "BertRCNN":
            from BertRCNN.BertRCNN import BertRCNN
            model = BertRCNN.from_pretrained(
                output_dir,
                config=config,
                rnn_hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                bidirectional=args.bidirectional,
                dropout=args.dropout
            )
        elif args.model_name == "BertCNNPlus":
            from BertCNNPlus.BertCNNPlus import BertCNNPlus
            filter_sizes = [int(val) for val in args.filter_sizes.split()]
            model = BertCNNPlus.from_pretrained(
                output_dir,
                config=config,
                n_filters=args.filter_num,
                filter_sizes=filter_sizes
            )
        elif args.model_name == "BertDPCNN":
            from BertDPCNN.BertDPCNN import BertDPCNN
            model = BertDPCNN.from_pretrained(
                output_dir,
                config=config,
                filter_num=args.filter_num
            )

        model.to(device)

        # 损失函数准备
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # test the model
        test_loss, test_acc, test_report, test_auc, all_idx, all_labels, all_preds = evaluate_save(
            model, test_dataloader, criterion, device, label_list)
        logger.info("==========================================》》》 Test 《《《=======================================")
        logger.info(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc*100: .3f} % | AUC:{test_auc}')

        for label in label_list:
            logger.info('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
                label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
        print_list = ['macro avg', 'weighted avg']

        for label in print_list:
            logger.info('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
                label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
