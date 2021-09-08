"""
该文件主要是测试ELProcess类，
将知识库中的文本和query文本进行连接，
然后构建正负样本。
"""
import json
import random


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

    def get_result(self, lines):
        examples = []
        for i,line in enumerate(lines):
            line = eval(line)
            text = line['text'].lower()
            for mention_data in line['mention_data']:
                word = mention_data['mention'].lower()
                kb_id = mention_data['kb_id']
                start_id = int(mention_data['offset'])
                end_id = start_id+len(word)-1
                # print((kb_id, word, start_id, end_id))
                rel_texts = self.get_text_pair(word, kb_id, text)
                for i,rel_text in enumerate(rel_texts):
                    print('text：', rel_text)
                    print('entity_label：', (kb_id, word, start_id, end_id))
                    if i == 0:
                        print('seq_label：', 1)
                    else:
                        print('seq_label：', 0)

            if i == 1:
                break

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



if __name__ == '__main__':
    elProcessor = ELProcessor()
    data_path = './data/ccks2019'
    train_data = data_path + '/train.json'
    elProcessor.get_result(elProcessor.read_json(train_data))
    # print(elProcessor.get_info('10010'))