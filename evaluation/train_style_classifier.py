import os 
import time
import random
import argparse
import json

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import RobertaTokenizer, RobertaModel, RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from roberta_style_classifier import StyleClassifier




class TransferDataset(Dataset):

    def __init__(self, data):
        self.texts = list(data["text"])
        self.target_labels = list(data["target_label"]) if "target_label" in data else list(data["target_style"])
        self.labels = list(data["label"]) if "label" in data else self.target_labels
        
        self.original_texts = list(data["original_text"]) if "original_text" in data else self.texts    # evaluation
        self.reference_texts = list(data["reference_text"]) if "reference_text" in data else self.texts # evaluation
        self.transferred_texts = list(data["transferred_text"]) if "transferred_text" in data else self.texts
        # self.insertion_labels = list(data["insertion_label"]) if "insertion_label" in data else self.labels # delete and insert

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # return {"text": self.texts[idx], "label": self.labels[idx], "target_label": self.target_labels[idx]}
        return {"text": self.texts[idx], "label": self.labels[idx], "target_label": self.target_labels[idx], "original_text": self.original_texts[idx],\
                "transferred_text": self.transferred_texts[idx], "reference_text": self.reference_texts[idx]} #"insertion_label": self.insertion_labels[idx]}



def get_dataset_lst(data_path, mode=None):

    with open(data_path, "r") as input_file:
        dataset_dict = json.load(input_file)
                
    labeled_texts = {}
    for key, value  in dataset_dict.items():
        if len(labeled_texts) == 0:
            labeled_texts["text"], labeled_texts["label"], labeled_texts["target_label"] = [], [], []
            if mode == "pretrain":
                labeled_texts["original_text"] = []
                labeled_texts["insertion_label"] = [] #delete and insert

            if mode == "test":
                labeled_texts["reference_text"] = []

        labeled_texts["text"].append(value["text"])

        labeled_texts["label"].append(value["label"])
        labeled_texts["target_label"].append(value["target_label"])

        if mode == "pretrain":
            if "original_text" in value:
                labeled_texts["original_text"].append(value["original_text"])
            else:
                labeled_texts["original_text"].append(value["text"])
            
            if "insertion_label" in value:
                labeled_texts["insertion_label"].append(value["insertion_label"])

        if mode == "test":
            labeled_texts["reference_text"].append(value["reference_text"])

    return labeled_texts



def train(args):
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    args.device = "cuda" if torch.cuda.is_available() else "cpu"


    model = StyleClassifier.from_pretrained('roberta-base',style_num=args.num_class).to(args.device)
    model.save_pretrained(os.path.join(args.save_dir, "roberta_base"))
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenizer.save_pretrained(os.path.join(args.save_dir, "roberta_tokenizer"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #pretrain_epoch = 0

    # if config.model_dir is not None:
    #     state = torch.load(config.model_dir)
    #     optimizer.load_state_dict(state["optim"])
    #     #styleClassifier.load_state_dict(state['model'])
    #     pretrain_epoch = state["epoch"] + 1

    train_set = get_dataset_lst(args.train_data_path, mode="train")
    train_set = TransferDataset(train_set)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)

    # test_set = get_dataset_lst(args.test_data_path, mode="test")
    # test_set = TransferDataset(test_set)
    # test_loader = DataLoader(test_set, shuffle=True, batch_size=args.batch_size)
    #test_classifier(config, styleClassifier, tokenizer, test_loader)

    step = 0
    for epoch in range(args.train_epochs):

        
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            step += 1
            start = time.time()
            style_labels, inputs = batch["label"], batch["original_text"]

            tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
            input_ids, attention_mask = tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]#, tokenized_inputs["token_type_ids"]
            batch_size, seq_length = input_ids.shape



            outputs = model(
                input_ids=input_ids.long().to(args.device), 
                attention_mask=attention_mask.long().to(args.device),
                # token_type_ids=token_type_ids.long().to(config.device),
                target_labels=style_labels.long().to(args.device))

            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.logging_per_steps == 0:

                end = time.time()
                print("epoch:{}\t batch:{}\t loss:{}\t, {}/{}, {}%\t time:{} sec".format(epoch, batch_idx, loss, batch_idx, len(train_loader), float(batch_idx / len(train_loader)), end-start))
                #test_classifier(config, styleClassifier, tokenizer, test_loader)

                # test_classifier(config, model, tokenizer, test_loader)

            torch.cuda.empty_cache()

        if epoch % args.saving_per_epochs == 0:
            pretrained_params = {"optim":optimizer.state_dict(), "epoch":epoch}
            # pretrained_optim_filename = os.path.join(config.pretrained_dir, "{}_classifier_epoch_{}.pth".format(config.datasource, i))
            pretrained_model_filename = os.path.join(args.save_dir, "new_roberta_{}_classifier_epoch_{}".format(args.datasource, epoch))
            model.save_pretrained(pretrained_model_filename)


    print("finish training style classifier.")
        
def test(args, model=None, tokenizer=None):
    if model is None:
        model = StyleClassifier.from_pretrained(args.model_path, style_num=args.num_class)
    if tokenizer is None:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()


    test_set = get_dataset_lst(args.test_data_path, mode="test")
    test_set = TransferDataset(test_set)
    test_data_loader = DataLoader(test_set, shuffle=True, batch_size=args.batch_size)
    #test_classifier(config, styleClassifier, tokenizer, test_loader)


    print("evaluating {}".format(args.test_data_path))
    correct, last_idx = 0, 0

    for batch_idx, batch in enumerate(test_data_loader):

        style_labels, inputs = batch["label"], batch["text"]

        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
        input_ids, attention_mask = tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"] #, tokenized_inputs["token_type_ids"]
        batch_size, seq_length = input_ids.shape
        
        # if seq_length > 50:
        #     continue

        with torch.no_grad():
            outputs = model(input_ids.long().to(device))

        predictions = outputs["logits"].argmax(dim=-1)
        correct += predictions.eq(style_labels.to(device)).sum().item()

    
    print("acc:{}/{}, {}%".format(correct, len(test_set), float(correct / len(test_set)) * 100))
    return correct * 1. / len(test_set)




if __name__ == '__main__':
    parser = argparse.ArgumentParser("train a style classifier.")

    parser.add_argument('--datasource', type=str, default="yelp")
    parser.add_argument('--random_seed',type=int, default=0)
    parser.add_argument('--evaluate', action='store_true')

    parser.add_argument('--model_path', type=str, default="../data/yelp/train_data_new.json")
    parser.add_argument('--tokenizer_path', type=str, default="../data/yelp/test_data.json")

    parser.add_argument('--train_epochs',type=int, default=5)
    parser.add_argument('--batch_size',type=int, default=8)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--train_data_path', type=str, default="../data/yelp/train_data_new.json")
    parser.add_argument('--test_data_path', type=str, default="../data/yelp/test_data.json")
    parser.add_argument('--pretrained_dir', type=str, default="./saved_models/")
    parser.add_argument('--save_dir', type=str, default="/saved_models/yelp_roberta_cls/")
    parser.add_argument('--model_dir', type=str, default=None)    
    parser.add_argument('--saving_per_epochs', type=int, default=1)
    parser.add_argument('--logging_per_steps', type=int, default=1000)

    args = parser.parse_args()
    
    if args.evaluate:
        test(args)
    else:
        train(args)
