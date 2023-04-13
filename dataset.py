import os
import json
from torch.utils.data import DataLoader, Dataset


class TransferDataset(Dataset):

    def __init__(self, data):
        self.texts = list(data["text"])
        self.target_labels = list(data["target_label"]) if "target_label" in data else list(data["target_style"])
        self.labels = list(data["label"]) if "label" in data else self.target_labels    # evaluation
        self.original_texts = list(data["original_text"]) if "original_text" in data else self.texts    # evaluation
        self.reference_texts = list(data["reference_text"]) if "reference_text" in data else self.texts # evaluation
        self.transferred_texts = list(data["transferred_text"]) if "transferred_text" in data else self.texts
        #print("texts:{}\t labels:{}\t target labels:{} original_texts:{}")
        print("finished loading {} data".format(len(self.texts)))
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        return {"text": self.texts[idx], "label": self.labels[idx], "target_label": self.target_labels[idx],\
                     "original_text": self.original_texts[idx], "reference_text": self.reference_texts[idx], "transferred_text": self.transferred_texts[idx]}

    def get_batches(self, batch_size, drop_last=False):

        batched_dataset = []
        batch_idx = 0
        num_batches = len(self.texts) // batch_size
        
        while batch_idx + batch_size < len(self): # drop last
            batched_dataset.append(self[batch_idx: batch_idx+batch_size])
            batch_idx += batch_size

        if not drop_last:
            batched_dataset.append(self[num_batches * batch_size :])
        return batched_dataset



def get_dataset_lst(data_path, mode="train"):
    
    with open(data_path, "r") as input_file:
        data_dict = json.load(input_file)
        
    labeled_texts = {
        "text": [],
        "label": [],
        "target_label": []
    }

    if mode == "pretrain":
        labeled_texts["original_text"] = []
        labeled_texts["insertion_label"] = [] #delete and insert
        
    elif mode == "test":
        labeled_texts["reference_text"] = []
        

    for key, value in data_dict.items():        
        
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



    
