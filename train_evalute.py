# coding=utf-8
import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time

from transformers import BertTokenizer, BertConfig


from Utils.utils import classifiction_metric
from Utils.load_datatsets import load_data
from Utils.logger import get_logger

logger = get_logger(__name__)

def train(args, epoch_num, n_gpu, model, tokenizer, train_dataloader, dev_dataloader,
          optimizer, scheduler, criterion, gradient_accumulation_steps, device, label_list,
          output_dir, log_dir, print_step, early_stop, do_test_after_per_epoch=True):
    """ 模型训练过程
    Args:
        args: 所有参数
        epoch_num: epoch 数量
        n_gpu: 使用的 gpu 数量
        train_dataloader: 训练数据的Dataloader
        dev_dataloader: 测试数据的 Dataloader
        optimizer: 优化器
        scheduler: 学习率调度
        criterion： 损失函数定义
        gradient_accumulation_steps: 梯度积累
        device: 设备，cuda， cpu
        label_list: 分类的标签数组
        output_dir: 用于保存 Bert 模型
        log_dir: tensorboard 读取的日志目录，用于后续分析
        print_step: 多少步保存一次模型，日志等信息
        early_stop: 提前终止
        do_test_after_per_epoch: 是否每个 epoch 进行测试, 默认为 False
    """

    

    early_stop_times = 0

    writer = SummaryWriter(
        log_dir=log_dir + '/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))


    best_dev_loss = float('inf')
    best_auc = 0
    best_acc = 0
    best_weighted_f1_for_dev = 0
    metric_info_for_test = []

    global_step = 0
    for epoch in range(int(epoch_num)):

        if early_stop_times >= early_stop:
            break

        print(f'======================================= Epoch: {epoch+1:02} ==========================================')

        epoch_loss = 0

        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            _, input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask, labels=None)
            loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

            """ 修正 loss """
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            train_steps += 1

            loss.backward()

            # 用于画图和分析的数据
            epoch_loss += loss.item()
            preds = logits.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)
            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)

            if (step + 1) % gradient_accumulation_steps == 0:
                # 更新模型参数
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % print_step == 0 and global_step != 0:

                    """ 打印Train此时的信息 """
                    train_loss = epoch_loss / train_steps
                    train_acc, train_report, train_auc = classifiction_metric(all_preds, all_labels, label_list)

                    dev_loss, dev_acc, dev_report, dev_auc = evaluate(model, dev_dataloader, criterion, device, label_list)

                    c = global_step // print_step
                    writer.add_scalar("loss/train", train_loss, c)
                    writer.add_scalar("loss/dev", dev_loss, c)

                    writer.add_scalar("acc/train", train_acc, c)
                    writer.add_scalar("acc/dev", dev_acc, c)

                    writer.add_scalar("auc/train", train_auc, c)
                    writer.add_scalar("auc/dev", dev_auc, c)

                    logger.info(f"global_step // print_step: {global_step} // {print_step}  = {c}")
                    logger.info(f"Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")
                    logger.info(f"Train Accuracy: {train_acc * 100: .4f} %, Dev Accuracy: {dev_acc * 100: .4f} %")
                    logger.info(f"Train AUC: {train_auc:.4f}, Dev AUC: {dev_auc:.4f}")

                    for label in label_list + ['macro avg', 'weighted avg']:
                        writer.add_scalar(f"{label}: f1/train", train_report[label]['f1-score'], c)
                        writer.add_scalar(f"{label}: f1/dev", dev_report[label]['f1-score'], c)
                        logger.info(f"Train {label} F1-Score: {train_report[label]['f1-score'] * 100:.4f}%, Dev {label} F1-Score: {dev_report[label]['f1-score'] * 100:.4f}%")

                    # # 以 acc 取优
                    # if dev_acc > best_acc:
                    #     best_acc = dev_acc
                    # 以 weighted_f1 取优
                    if dev_report['weighted avg']['f1-score'] > best_weighted_f1_for_dev:
                        best_weighted_f1_for_dev = dev_report['weighted avg']['f1-score']

                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_vocabulary(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        early_stop_times = 0
                    else:
                        early_stop_times += 1

        if do_test_after_per_epoch:
            print(f"==========================================》》》 Test after epoch_{epoch + 1} 《《《=======================================")
            # # 保存每个 epoch 的模型
            # model_to_save = (
            #     model.module if hasattr(model, "module") else model
            # )  # Take care of distributed/parallel training
            # epoch_model_save_path = os.path.join(output_dir, f"epoch_{epoch+1}")
            # model_to_save.save_pretrained(epoch_model_save_path)
            # tokenizer.save_vocabulary(epoch_model_save_path)
            # logger.info("Saving epoch model checkpoint to %s", epoch_model_save_path)
            # Test
            test_dataloader, _ = load_data(args.data_dir, tokenizer, args.max_seq_length,
                                           args.test_batch_size, "test", label_list)
            # test the epoch model
            test_loss, test_acc, test_report, test_auc = evaluate(model, test_dataloader, criterion, device, label_list)
            print(f'\t Loss: {test_loss:.3f} | Acc: {test_acc * 100:.4f}% | AUC: {test_auc:.4f}')

            for label in label_list + ['macro avg', 'weighted avg']:
                print('\t {}: Precision: {:.4f}% | Recall: {:.4f}% | F1 Score: {:.4f}%'.format(
                    label, test_report[label]['precision'] * 100, test_report[label]['recall'] * 100,
                           test_report[label]['f1-score'] * 100))

            metric_info_for_test.append({
                'epoch': epoch+1,
                'auc': test_auc,
                'accuracy': test_acc * 100,
                'P_macro_avg': test_report['macro avg']['precision'] * 100,
                'R_macro_avg': test_report['macro avg']['recall'] * 100,
                'F1_macro_avg': test_report['macro avg']['f1-score'] * 100,
                'P_weighted_avg': test_report['weighted avg']['precision'] * 100,
                'R_weighted_avg': test_report['weighted avg']['recall'] * 100,
                'F1_weighted_avg': test_report['weighted avg']['f1-score'] * 100
            })

    with open(os.path.join(output_dir, 'metric_info_for_test.json'), 'w') as f:
        json.dump(metric_info_for_test, f, ensure_ascii=False, indent=4)

    writer.close()
                    

def evaluate(model, dataloader, criterion, device, label_list):

    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    epoch_loss = 0

    for _, input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Eval"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        epoch_loss += loss.mean().item()

    acc, report, auc = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss/len(dataloader), acc, report, auc


def evaluate_save(model, dataloader, criterion, device, label_list):

    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    all_idxs = np.array([], dtype=int)

    epoch_loss = 0

    for idxs, input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Eval"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        idxs = idxs.detach().cpu().numpy()
        all_idxs = np.append(all_idxs, idxs)

        epoch_loss += loss.mean().item()

    acc, report, auc = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss/len(dataloader), acc, report, auc, all_idxs, all_labels, all_preds
