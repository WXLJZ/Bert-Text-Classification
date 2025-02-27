# coding=utf-8

import argparse


def get_args(data_dir, output_dir, cache_dir, model_name_or_path, log_dir):

    parser = argparse.ArgumentParser(description='BERT Baseline')

    parser.add_argument("--model_name", default="BertCNNPlus",
                        type=str, help="the name of model ")
    parser.add_argument("--save_name", default="BertCNNPlus",
                        type=str, help="the name file of model")

    # 文件路径：数据目录， 缓存目录
    parser.add_argument("--data_dir",
                        default=data_dir,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default=output_dir + "BertCNNPlus/",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--cache_dir",
                        default=cache_dir + "BertCNNPlus/",
                        type=str,
                        help="缓存目录，主要用于模型缓存")

    parser.add_argument("--log_dir",
                        default=log_dir + "BertCNNPlus/",
                        type=str,
                        help="日志目录，主要用于 tensorboard 分析")

    parser.add_argument("--model_name_or_path",
                        default=model_name_or_path,
                        type=str)

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="随机种子 for initialization")

    # 文本预处理参数
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # 训练参数
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--dev_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for dev.")
    parser.add_argument("--test_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for test.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
    # optimizer 参数
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam 的 学习率")

    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")

    # 梯度累积
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--print_step',
                        type=int,
                        default=200,
                        help="多少步进行模型保存以及日志信息写入")

    #CNN 参数
    parser.add_argument("--filter_num", default=50,
                        type=int, help="filter 的数量")
    parser.add_argument("--filter_sizes", default="1 2 3 4 5 6 7 8 9 10 11",
                        type=str, help="filter 的 size")

    parser.add_argument("--early_stop", type=int, default=30,
                        help="提前终止，多少次dev loss 连续增大，就不再训练")

    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu 的设备id")
    config = parser.parse_args()

    return config
