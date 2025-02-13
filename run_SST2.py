# coding=utf-8
from main import main


if __name__ == "__main__":
    model_name_list = ['BertOrigin', 'BertCNN', 'BertLSTM', 'BertATT', 'BertRCNN', 'BertCNNPlus', 'BertDPCNN']
    label_list = ['0', '1']
    data_dir = "./data/SST2"
    output_dir = "./sst2_output/"
    cache_dir = "./sst2_cache/"
    log_dir = "./sst2_log/"

    # bert-base
    model_name_or_path = "/home/XXXX/models/bert-base-uncased"

    for model_name in model_name_list:
        if model_name == "BertOrigin":
            from BertOrigin import args

        elif model_name == "BertCNN":
            from BertCNN import args

        elif model_name == 'BertLSTM':
            from BertLSTM import args

        elif model_name == "BertATT":
            from BertATT import args

        elif model_name == "BertRCNN":
            from BertRCNN import args

        elif model_name == "BertCNNPlus":
            from BertCNNPlus import args

        elif model_name == "BertDPCNN":
            from BertDPCNN import args

        args = args.get_args(
            data_dir=data_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            model_name_or_path=model_name_or_path,
            log_dir=log_dir
        )

        main(args, args.save_name, label_list)
        print("================================================================================================")
        print()
        print()