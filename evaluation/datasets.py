from torch.utils.data import Dataset

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

