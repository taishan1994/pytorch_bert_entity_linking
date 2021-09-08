import os
import logging
import time
import numpy as np
import pickle
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW

import el_config
from el_preprocess import BertFeature
import el_dataset
import el_models
import utils
import my_jieba
from utils import tokenization

logger = logging.getLogger(__name__)
args = el_config.Args().get_parser()
utils.utils.set_seed(args.seed)
utils.utils.set_logger(os.path.join(args.log_dir, 'main.log'))


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model = el_models.BertForEntityLinking(args)
        # self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        if train_loader:
            self.optimizer, self.scheduler = self.configure_optimizers()
        self.model.to(self.device)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        t_total = len(self.train_loader) * self.args.train_epochs
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=t_total)

        return optimizer, scheduler

    def load_ckp(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    """
    def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
        tmp_checkpoint_path = checkpoint_path
        torch.save(state, tmp_checkpoint_path)
        if is_best:
            tmp_best_model_path = best_model_path
            shutil.copyfile(tmp_checkpoint_path, tmp_best_model_path)
    """

    def train(self):
        total_step = len(self.train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = 100
        best_dev_micro_f1 = 0.0
        self.model.zero_grad()
        for epoch in range(self.args.train_epochs):
            for train_step, train_data in enumerate(self.train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                seq_labels = train_data['seq_labels'].to(self.device)
                entity_labels = train_data['entity_labels'].to(self.device)
                self.model.zero_grad()
                train_outputs, loss = self.model(token_ids, attention_masks, token_type_ids, seq_labels, entity_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                logger.info(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                # 直接训练完再预测
                # if global_step % eval_step == 0:
                #     dev_loss, dev_outputs, dev_targets = self.dev()
                #     accuracy, recision, recall, micro_f1 = self.get_metrics(dev_outputs, dev_targets)
                #     logger.info(
                #         "【dev】 loss：{:.6f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} micro_f1：{:.4f}".format(dev_loss, accuracy, precision, recall, micro_f1))
                #     if micro_f1 > best_dev_micro_f1:
                #         logger.info("------------>保存当前最好的模型")
                #         checkpoint = {
                #             'epoch': epoch,
                #             'loss': dev_loss,
                #             'state_dict': self.model.state_dict(),
                #         }
                #         best_dev_micro_f1 = micro_f1
                #         checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                #         self.save_ckp(checkpoint, checkpoint_path)
                if global_step % 5000 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'loss': loss.item(),
                        'state_dict': self.model.state_dict(),
                    }
                    checkpoint_path = os.path.join(self.args.output_dir, '{}.pt'.format(str(global_step)))
                    self.save_ckp(checkpoint, checkpoint_path)

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                seq_labels = dev_data['seq_labels'].to(self.device)
                entity_labels = dev_data['entity_labels'].to(self.device)
                outputs, loss = self.model(token_ids, attention_masks, token_type_ids,
                                           seq_labels, entity_labels)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                seq_labels = np.argmax(seq_labels.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(seq_labels.tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        model, epoch, loss = self.load_ckp(model, checkpoint_path)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            total_step = len(self.test_loader)
            for test_step, test_data in enumerate(self.test_loader):
                print(test_step, total_step)
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                seq_labels = test_data['seq_labels'].to(self.device)
                entity_labels = test_data['entity_labels'].to(self.device)
                outputs, loss = model(token_ids, attention_masks, token_type_ids,
                                      seq_labels, entity_labels)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                seq_labels = np.argmax(seq_labels.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(seq_labels.tolist())

        return total_loss, test_outputs, test_targets

    def convert_example_to_feature(self0,
                                   text_b,
                                   start,
                                   end,
                                   ids,
                                   tokenizer,
                                   tokenization,
                                   subject_id_with_info,
                                   args):
        features = []
        for t_id in ids:
            if t_id in subject_id_with_info:
                info = subject_id_with_info[t_id]
                text_a_list = []
                for kg in info['data']:
                    # print(kg)
                    if kg['object'][-1] != '。':
                        text_a_list.append("{}，{}。".format(kg['predicate'],kg['object']))
                    else:
                        text_a_list.append("{}，{}".format(kg['predicate'], kg['object']))

                text_a = "".join(text_a_list)
                text_a = tokenization.BasicTokenizer().tokenize(text_a)
                encode_dict = tokenizer.encode_plus(text=text_a,
                                                    text_pair=text_b,
                                                    max_length=args.max_seq_len,
                                                    padding='max_length',
                                                    truncation='only_first',
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
                token_ids = encode_dict['input_ids']
                attention_masks = encode_dict['attention_mask']
                token_type_ids = encode_dict['token_type_ids']

                offset = token_type_ids[0].tolist().index(1)  # 找到1最先出现的位置
                entity_ids = [0] * args.max_seq_len
                # print(start)
                # print(end)
                # print(offset)
                start_id = offset + start
                end_id = offset + end
                # print(start_id)
                # print(end_id)
                # print(tokenizer.convert_ids_to_tokens(token_ids[0]))
                # print(tokenizer.convert_ids_to_tokens(token_ids[0][start_id:end_id+1]))
                # print(len(token_ids[0]))
                # print(start_id, end_id)
                for i in range(start_id, end_id):
                    entity_ids[i] = 1
                entity_ids = torch.tensor(entity_ids, requires_grad=False).unsqueeze(0)
                features.append(
                    (
                        token_ids,
                        attention_masks,
                        token_type_ids,
                        entity_ids,
                        info['subject_id'],
                        info['subject'],
                        info['type'],
                        "".join(text_a_list),
                    )
                )
        return features


    def predict(self,
                checkpoint_path,
                text,
                args,
                tokenizer,
                tokenization,
                entities,
                entity_to_ids,
                subject_id_with_info,
                ):
        model = self.model
        model, epoch, loss = self.load_ckp(model, checkpoint_path)
        model.eval()
        model.to(self.device)
        # 先提取text中的实体，这里结合实体库利用jieba分词进行
        text = text.lower()
        words = my_jieba.lcut(text, cut_all=False)
        # text_b=['《', '仙剑奇侠', '三', '》', '紫萱', '为', '保护', '林业平', '被迫', '显出', '原型']
        # result中是一个元组，第一维表示该实体名，第二位是在知识库中的subject_id，第三位是分数,
        # 第四位是真实名，第五位是类型，第六位是描述
        result = []
        NIL_list = []
        with torch.no_grad():
            for word in words:
                # 如果该词是一个候选实体，那么我们从知识库中找到其subject_id
                if word in entities:
                    # print(word)
                    tmp_res = []
                    ids = entity_to_ids[word]
                    if len(ids) == 1 and ids[-1] == 'NIL':
                        NIL_list.append(word)
                    else:
                        # 在文本中找到该实体的起始和结束位置,这里我们只找第一次出现的位置就行了
                        # 这里我们要合并这两个分词的结果
                        ind = text.index(word)
                        start_ = tokenization.BasicTokenizer().tokenize(text[:ind])
                        word_ = tokenization.BasicTokenizer().tokenize(word)
                        end_ = tokenization.BasicTokenizer().tokenize(text[ind+len(word):])
                        start = len(start_)
                        end = start+len(word_)
                        text_b = start_ + word_ + end_
                        # print(word)
                        # print(text_b)
                        features = self.convert_example_to_feature(
                            text_b,
                            start,
                            end,
                            ids,
                            tokenizer,
                            tokenization,
                            subject_id_with_info,
                            args,
                        )
                        # print('features:', features)
                        if len(features) != 0:
                            for feature in features:
                                logit = model(
                                    feature[0].to(self.device),
                                    feature[1].to(self.device),
                                    feature[2].to(self.device),
                                    None,
                                    feature[3].to(self.device),
                                )
                                # print(logit)
                                sigmoid = nn.Sigmoid()
                                logit = sigmoid(logit)
                                pred = logit.cpu().detach().numpy()[0][1]
                                # print(pred)
                                tmp_res.append(
                                    (
                                        word,
                                        feature[4],
                                        pred,
                                        feature[5],
                                        feature[6],
                                        feature[7],
                                    )
                                )
                            tmp_res = sorted(tmp_res, key=lambda x:x[2], reverse=True)
                            print(tmp_res)
                            result.append(tmp_res)
                        else:
                            continue
        return result, NIL_list


    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        precision = precision_score(targets, outputs)
        recall = recall_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        return accuracy, precision, recall, micro_f1

    def get_classification_report(self, outputs, targets):
        report = classification_report(targets, outputs)
        return report


if __name__ == '__main__':
    # train_out = pickle.load(open('./data/ccks2019/train.pkl', 'rb'))
    # train_features, train_callback_info = train_out
    # train_dataset = el_dataset.ELDataset(train_features)
    # train_sampler = RandomSampler(train_dataset)
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=args.train_batch_size,
    #                           sampler=train_sampler,
    #                           num_workers=2)
    #
    # dev_out = pickle.load(open('./data/ccks2019/test.pkl', 'rb'))
    # dev_features, dev_callback_info = dev_out
    # dev_dataset = el_dataset.ELDataset(dev_features)
    # dev_loader = DataLoader(dataset=dev_dataset,
    #                         batch_size=args.eval_batch_size,
    #                         num_workers=2)
    #
    # test_out = pickle.load(open('./data/ccks2019/test.pkl', 'rb'))
    # test_features, test_callback_info = dev_out
    # test_dataset = el_dataset.ELDataset(test_features)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=args.eval_batch_size,
    #                          num_workers=2)
    #
    # trainer = Trainer(args, train_loader, dev_loader, test_loader)
    trainer = Trainer(args, None, None, None)
    # 训练和验证
    # trainer.train()

    # 测试
    # test_start_time = time.time()
    # logger.info('========进行测试========')
    # checkpoint_path = './checkpoints/15000.pt'
    # total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    # accuracy, precision, recall, micro_f1 = trainer.get_metrics(test_outputs, test_targets)
    # logger.info(
    #     "【test】 loss：{:.6f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} micro_f1：{:.4f}".format(total_loss, accuracy, precision, recall, micro_f1))
    # report = trainer.get_classification_report(test_outputs, test_targets)
    # logger.info(report)
    # test_end_time = time.time()
    # print('预测耗时：{}s，平均每条耗时：{}s'.format(test_end_time-test_start_time,(test_end_time-test_start_time)/len(test_dataset)))

    # 预测
    checkpoint_path = './checkpoints/15000.pt'
    with open('./checkpoints/args.json','w') as fp:
        fp.write(json.dumps(vars(args)))
    my_jieba.load_userdict('./data/ccks2019/alias_and_subjects.txt')
    # 实体库
    with open('./data/ccks2019/alias_and_subjects.txt', 'r') as fp:
        entities = fp.read().strip().split('\n')
    # 实体对应的id
    with open('./data/ccks2019/entity_to_ids.json','r') as fp:
        entity_to_ids = json.loads(fp.read())
    # 实体id对应的描述
    with open('./data/ccks2019/subject_id_with_info.json','r') as fp:
        subject_id_with_info = json.loads(fp.read())
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir + 'vocab.txt')
    text = '《仙剑奇侠三》紫萱为保护林业平被迫显出原型'
    result, NIL_list = trainer.predict(checkpoint_path=checkpoint_path, text=text, args=args, tokenizer=tokenizer, tokenization=tokenization, entities=entities, entity_to_ids=entity_to_ids, subject_id_with_info=subject_id_with_info)
    for res in result:
        # print(res)
        for info in res: # 这里我们选择分数最高的打印
            # print(info)
            logger.info('====================================')
            logger.info('候选实体名：' + info[0])
            logger.info('知识库实体名：' + info[3])
            logger.info('知识库ID：' + info[1])
            logger.info('置信分数：' + str(info[2]))
            logger.info('类型：' + '、'.join(info[4]))
            logger.info('描述：' + info[5][:100] + '......')
            logger.info('====================================')
            break
    # print('找到实体，但实体库中并未存在相关信息：', NIL_list)


