import os
import json
import logging
import random
import pickle
import numpy as np

from transformers import BertTokenizer
import el_config
from utils import utils
from utils import tokenization

logger = logging.getLogger(__name__)
args = el_config.Args().get_parser()
utils.set_seed(args.seed)
utils.set_logger(os.path.join(args.log_dir, 'el_preprocess.log'))


class InputExample:
    def __init__(self, set_type, text, seq_label, entity_label):
        self.set_type = set_type
        self.text = text
        self.seq_label = seq_label
        self.entity_label = entity_label


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, seq_labels, entity_labels):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.seq_labels = seq_labels
        self.entity_labels = entity_labels


class ELProcessor:
    def __init__(self):
        with open('./data/ccks2019/entity_to_ids.json','r') as fp:
            self.entity_to_ids = json.loads(fp.read())
        with open('./data/ccks2019/subject_id_with_info.json','r') as fp:
            self.subject_id_with_info = json.loads(fp.read())

    def read_json(self, path):
        with open(path,'r') as fp:
            lines = fp.readlines()
        return lines

    def get_result(self, lines, set_type):
        examples = []
        for i,line in enumerate(lines):
            line = eval(line)
            text = line['text'].lower()
            for mention_data in line['mention_data']:
                word = mention_data['mention'].lower()
                kb_id = mention_data['kb_id']
                start_id = int(mention_data['offset'])
                end_id = start_id+len(word)-1
                rel_texts = self.get_text_pair(word, kb_id, text)
                for i, rel_text in enumerate(rel_texts):
                    if i == 0:
                        examples.append(InputExample(
                            set_type=set_type,
                            text=rel_text,
                            seq_label=1,
                            entity_label=(kb_id, word, start_id, end_id)
                        ))
                    else:
                        examples.append(InputExample(
                            set_type=set_type,
                            text=rel_text,
                            seq_label=0,
                            entity_label=(kb_id, word, start_id, end_id)
                        ))
        return examples

    def get_text_pair(self, word, kb_id, text):
        """
        用于构建正负样本对，一个正样本，三个负样本
        :return:
        """
        results = []
        if kb_id != 'NIL' and word in self.entity_to_ids:
            pos_example = self.get_info(kb_id) + '#;#' + text
            results.append(pos_example)
            ids = self.entity_to_ids[word]
            if 'NIL' in ids:
                ids.remove('NIL')
            ind = ids.index(kb_id)
            ids = ids[:ind] + ids[ind+1:]
            if len(ids) >= 3:
                ids = random.sample(ids, 3)
            for t_id in ids:
                info = self.get_info(t_id)
                neg_example  = info + '#;#' + text
                results.append(neg_example)
        return results


    def get_info(self, subject_id):
        """
        根据subject_id找到其描述文本，将predicate和object拼接
        :param subject_id:
        :return:
        """
        infos = self.subject_id_with_info[subject_id]
        data = infos['data']
        res = []
        for kg in data:
            if kg['object'][-1] != '。':
                res.append("{}，{}。".format(kg['predicate'],kg['object']))
            else:
                res.append("{}，{}".format(kg['predicate'], kg['object']))
        return "".join(res).lower()


def convert_bert_example(ex_idx, example: InputExample, tokenizer: BertTokenizer, max_seq_len):
    set_type = example.set_type
    raw_text = example.text
    seq_label = example.seq_label
    entity_label = example.entity_label # (subject_id,mention,start,end)
    # 文本元组
    text_a, text_b = raw_text.split('#;#')
    tokens_a = tokenization.BasicTokenizer().tokenize(text_a)

    # 将句子标签进行one-hot编码
    seq_final_label = [0, 0]
    if seq_label == 0:
        seq_final_label[0] = 1
    else:
        seq_final_label[1] = 1

    # 这里避免将英文切分开，这里使用tokenization里面的BasicTokenzier进行切分，
    # 切分之后要重新对实体的索引进行调整
    start = entity_label[2]
    end = entity_label[3]
    tokenizer_pre = tokenization.BasicTokenizer().tokenize(text_b[:start])
    tokenizer_label = tokenization.BasicTokenizer().tokenize(entity_label[1])
    tokenizer_post = tokenization.BasicTokenizer().tokenize(text_b[end+1:])
    real_label_start = len(tokenizer_pre)
    real_label_end = len(tokenizer_pre) + len(tokenizer_label)
    tokens_b = tokenizer_pre + tokenizer_label + tokenizer_post

    try:
        encode_dict = tokenizer.encode_plus(text=tokens_a,
                                            text_pair=tokens_b,
                                            max_length=max_seq_len,
                                            padding='max_length',
                                            truncation_strategy='only_first',
                                            return_token_type_ids=True,
                                            return_attention_mask=True)
    except Exception as e:
        print(e)
        print(tokens_a)
        print(tokens_b)
        return '出现错误了','400'
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    offset = token_type_ids.index(1) # 找到1最先出现的位置
    entity_ids = [0] * max_seq_len
    start_id = offset + real_label_start
    end_id = offset + real_label_end
    if end_id > max_seq_len:
        print('发生了不该有的截断')
        for i in range(start_id, max_seq_len):
            entity_ids[i] = 1
    else:
        for i in range(start_id, end_id):
            entity_ids[i] = 1
    callback_info = (text_b,)
    callback_entity_labels = (entity_label[1], offset)
    callback_info += (callback_entity_labels,)
    if ex_idx < 3:
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f"text: {raw_text}")
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f"entity_ids：{entity_ids}")
        logger.info(f"seq_label：{seq_final_label}")
        logger.info((tokenizer.convert_ids_to_tokens(token_ids[start_id:end_id + 1])))
    feature = BertFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        seq_labels=seq_final_label,
        entity_labels=entity_ids,

    )

    return feature, callback_info

def convert_examples_to_features(examples, max_seq_len, bert_dir):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')
    total = len(examples)
    for i, example in enumerate(examples):
        print(i, total)
        feature, tmp_callback = convert_bert_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )
        if tmp_callback == '400':
            continue
        if feature is None:
            continue

        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out

def split_train_test(examples, train_rate):
    total = len(examples)
    train_total = int(total * train_rate)
    test_total = total - train_total
    print('总共有数据：{}，划分后训练集：{}，测试集：{}'.format(total, train_total, test_total))
    random.shuffle(examples)
    train_examples = examples[:train_total]
    test_examples = examples[train_total:]
    return train_examples, test_examples

def get_out(processor, txt_path, args, mode):
    raw_examples = processor.read_json(txt_path)
    examples = processor.get_result(raw_examples, mode)
    for i, example in enumerate(examples):
        print(example.text)
        print(example.seq_label)
        print(example.entity_label)
        if i == 1:
            break
    train_examples, test_examples = split_train_test(examples, 0.7)
    train_out = convert_examples_to_features(train_examples, args.max_seq_len, args.bert_dir)
    test_out = convert_examples_to_features(test_examples, args.max_seq_len, args.bert_dir)
    return train_out, test_out



if __name__ == '__main__':
    args.max_seq_len = 256
    logger.info(vars(args))

    elprocessor = ELProcessor()

    train_out, test_out = get_out(elprocessor, os.path.join(args.data_dir, 'train.json'), args, 'train')
    with open(args.data_dir  + 'train.pkl', 'wb') as fp:
        pickle.dump(train_out, fp)
    with open(args.data_dir  + 'test.pkl', 'wb') as fp:
        pickle.dump(test_out, fp)
    # 由于只有训练数据，我们要对训练数据进行划分
    # dev_out = get_out(elprocessor, os.path.join(args.data_dir, 'dev.json'), args, 'dev')
    # test_out = get_out(elprocessor, os.path.join(args.data_dir, 'test.json'), args, 'test')
