import sys
from datetime import datetime
from os.path import join
from tqdm.auto import tqdm, trange
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import transformers
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Config
import numpy as np

import json
data = []
with open("train.txt","r",encoding="utf-8") as f:
    for i in f.readlines():
        line = json.loads(i)
        data.append(line)

def preprocess_conversation(data,model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    dialogue_list = []
    for conver in data:
        input_ids = [cls_id]
        for i in conver["conversation"]:
            input_ids += tokenizer.encode(i["utterance"], add_special_tokens=False)
            input_ids.append(sep_id)
        dialogue_list.append(input_ids)
    return dialogue_list

dialogue_list = preprocess_conversation(data,"shibing624/gpt2-dialogbot-base-chinese")


class MyDataset(Dataset):
    """
    GPT model dataset
    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


train_dataset = MyDataset(dialogue_list, 100)
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_fn,
    drop_last=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained("shibing624/gpt2-dialogbot-base-chinese")

model.to(device)

from  datetime import datetime
start_time = datetime.now()

epochs = 5
optimizer =  torch.optim.Adam(params=model.parameters(),lr = 0.0001)
losses = []
model.train()
for i in range(epochs):
    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # 梯度清零
        optimizer.zero_grad()
        outputs = model.forward(input_ids.to(device), labels=labels.to(device))
        logits = outputs.logits
        loss = outputs.loss
        losses.append(loss.mean().item())
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('test--1')
            print(np.mean(losses))

end_time = datetime.now()
print('waist time',end_time - start_time)

torch.save(model, 'new/pytorch_model.bin')

class GPTBot:
    def __init__(
            self,
            model_name_or_path,
            max_history_len=3,
            max_len=200
    ):
        self.tokenizer = BertTokenizerFast.from_pretrained('shibing624/gpt2-dialogbot-base-chinese')
        if 'new' in model_name_or_path:
            self.model = torch.load(model_name_or_path)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model .to(device)

        self.model.eval()
        self.history = []
        self.max_history_len = max_history_len
        self.max_len = max_len

    def predict(self, query, use_history=True):
        text_ids = self.tokenizer.encode(query, add_special_tokens=False)
        self.history.append(text_ids)
        input_ids = [self.tokenizer.cls_token_id]  # 每个input以[CLS]为开头
        if use_history:
            for history_id, history_utr in enumerate(self.history[-self.max_history_len:]):
                    input_ids.extend(history_utr)
                    input_ids.append(self.tokenizer.sep_token_id)
        else:
            input_ids.extend(text_ids)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0)
        response = []

        for _ in range(self.max_len):
            outputs = self.model(input_ids=input_ids.to(device)).cpu()
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            if next_token == self.tokenizer.sep_token_id:  #
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
        self.history.append(response)
        response_tokens = self.tokenizer.convert_ids_to_tokens(response)
        return "".join(response_tokens)


bot = GPTBot("shibing624/gpt2-dialogbot-base-chinese")
print(bot.predict("最近天气好好，想出去拍照片"))

bot = GPTBot("new/pytorch_model.bin")
print(bot.predict("最近天气好好，想出去拍照片"))
