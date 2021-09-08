from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn as nn
import torch
import torch.nn.functional as F
from el_preprocess import BertFeature
from el_dataset import ELDataset


class BertForEntityLinking(nn.Module):
    def __init__(self, args):
        super(BertForEntityLinking, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.criterion = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(out_dims, args.num_tags)


    def forward(self, token_ids, attention_masks, token_type_ids, seq_labels, entity_labels):
        bert_outputs = self.bert(
            input_ids = token_ids,
            attention_mask = attention_masks,
            token_type_ids = token_type_ids,
        )
        #  CLS的向量
        token_out = bert_outputs[0] #[2,256,768)]
        seq_out = bert_outputs[1] #[2, 768)]

        batch_out = []
        for t_out, t_mask, s_out in zip(token_out, entity_labels, seq_out):
            t_mask = t_mask == 1 #[768]
            entity_out = t_out[t_mask] #[256,768]

            out = torch.cat([entity_out, s_out.unsqueeze(0)], dim=0).unsqueeze(0) #[1,3,768]

            out = F.adaptive_max_pool1d(out.transpose(1,2).contiguous(), output_size=1) #[1,768,1]
            out = out.squeeze(-1) #[1,768]
            batch_out.append(out)
        batch_out = torch.cat(batch_out, dim=0)
        batch_out = self.linear(batch_out)
        if seq_labels is None:
            return batch_out
        batch_out = self.dropout(batch_out)
        loss = self.criterion(batch_out, seq_labels.float())
        return batch_out, loss

if __name__ == '__main__':
    class Args:
        bert_dir = '../model_hub/chinese-bert-wwm-ext/'
        num_tags = 2
        eval_batch_size = 4
    args = Args()
    import pickle
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir+'vocab.txt')
    test_out = pickle.load(open('./data/ccks2019/test.pkl', 'rb'))
    test_features, test_callback_info = test_out
    test_dataset = ELDataset(test_features)
    # for data in test_dataset:
        # text = tokenizer.convert_ids_to_tokens(data['token_ids'])
        # print(text)
    #     print(data['attention_masks'])
    #     print(data['token_type_ids'])
    #     print(data['seq_labels'])
    #     print(data['entity_labels'])
    #     break

    args.eval_batch_size = 4
    test_sampler = RandomSampler(test_dataset)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.eval_batch_size,
                             sampler=test_sampler,
                             num_workers=2)
    device = torch.device("cuda:0")
    model = BertForEnttityLinking(args)
    model.to(device)
    for step, test_data in enumerate(test_loader):
        # print(test_data['token_ids'].shape)
        # print(test_data['attention_masks'].shape)
        # print(test_data['token_type_ids'].shape)
        # print(test_data['seq_labels'])
        # print(test_data['entity_labels'])
        for key in test_data:
            test_data[key] = test_data[key].to(device)
        _, loss = model(test_data['token_ids'],
              test_data['attention_masks'],
              test_data['token_type_ids'],
              test_data['seq_labels'],
              test_data['entity_labels'])
        print(loss.item())
        break