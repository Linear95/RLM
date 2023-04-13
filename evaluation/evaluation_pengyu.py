import re
import os
import glob
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

from collections import defaultdict

from data_loader import *
from yelp_roberta_classifier import StyleClassifier
# from imdb_xlnet_classifier import StyleClassifier
# from imdb_bert_classifier import StyleClassifier

parser = argparse.ArgumentParser("bleu evaluation")
parser.add_argument("--result-path", type=str, default=None, help="result path.")
parser.add_argument("--model-dir",type=str, default="./saved_models/roberta_yelp_classifier_epoch_1")
parser.add_argument("--batch-size", type=int, default=1, help="batch size.")

# cls params
parser.add_argument('--cls-hidden-size', type=int, default=128)
parser.add_argument('--num-cls-layers', type=int, default=6)
parser.add_argument('--embedding-size', type=int, default=300)
parser.add_argument('--vocab-size', type=int, default=30522)
parser.add_argument('--dropout-rate', type=float, default=0.4)
parser.add_argument('--num-class', type=int, default=2)

parser.add_argument("--random-seed", type=int, default=0, help="random seed.")

config = parser.parse_args()
config.device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(config.random_seed)
torch.cuda.manual_seed_all(config.random_seed)
np.random.seed(config.random_seed)
random.seed(config.random_seed)

torch.backends.cudnn.deterministic = True

# # bleu evaluation

def compute_bleu(reference_texts, transferred_texts):
    return sentence_bleu(reference_texts, transferred_texts, weights=(0.5, 0.5)) * 100

def evaluate_bleu(evaluation_files):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    reference_texts, transferred_texts, original_texts = evaluation_files["reference_text"], evaluation_files["transferred_text"], evaluation_files["text"]

    reference_texts = [re.sub(r"[,.?\-&`:'()!]+"," ", text.strip()) for text in reference_texts]
    original_texts = [re.sub(r"[,.?\-&`:'()!]+"," ", text.strip()) for text in original_texts]
    transferred_texts = [re.sub(r"[,.\-&`:'?()!]+"," ", text.strip().strip("[CLS]").strip("[SEP]")) for text in transferred_texts]

    print(transferred_texts[:20])

    bleu_outputs = {}
    ref_bleu_scores, self_bleu_scores = [], []

    for reference_text, transferred_text in zip(reference_texts, transferred_texts):

        reference_text, transferred_text = [reference_text.split(" ")], transferred_text.split(" ")
        ref_bleu_score = compute_bleu(reference_text, transferred_text)
        ref_bleu_scores.append(ref_bleu_score)

    for original_text, transferred_text in zip(original_texts, transferred_texts):

        original_text, transferred_text = [original_text.split(" ")], transferred_text.split(" ")
        self_bleu_score = compute_bleu(original_text, transferred_text)
        self_bleu_scores.append(self_bleu_score)

    avg_ref_bleu_score = sum(ref_bleu_scores) / len(ref_bleu_scores)
    avg_self_bleu_score = sum(self_bleu_scores) / len(self_bleu_scores)

    bleu_outputs = {"ref_bleu": avg_ref_bleu_score, "self_bleu": avg_self_bleu_score}
    
    return bleu_outputs

def evaluate_cls(evaluation_files, pretrained_cls, tokenizer, config):

    pretrained_cls.eval()
    pretrained_cls.to(config.device)
        
    test_set = TransferDataset(evaluation_files)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    trans_correct, ref_correct, original_correct = 0, 0, 0

    for batch_idx, batch in enumerate(test_loader):

        style_labels = batch['target_label'].to(config.device)
        original_labels = torch.ones(size=style_labels.shape).to(config.device) - style_labels

        transferred_inputs = tokenizer(batch['transferred_text'], padding=True, return_tensors="pt")
        transferred_input_ids = torch.tensor(transferred_inputs['input_ids']).long().to(config.device)
    
        reference_inputs = tokenizer(batch["reference_text"], padding=True, return_tensors="pt")
        reference_input_ids = torch.tensor(reference_inputs['input_ids']).long().to(config.device)
        
        original_inputs = tokenizer(batch["text"], padding=True, return_tensors="pt")
        original_input_ids = torch.tensor(original_inputs['input_ids']).long().to(config.device)

        with torch.no_grad():
            transferred_outputs = pretrained_cls(transferred_input_ids)

        with torch.no_grad():
            reference_outputs = pretrained_cls(reference_input_ids)
    
        with torch.no_grad():
            original_outputs = pretrained_cls(original_input_ids)
            

        transferred_outputs = transferred_outputs["logits"].argmax(dim=-1)
        trans_correct += transferred_outputs.eq(style_labels.to(config.device)).sum().item()
    
        reference_outputs = reference_outputs["logits"].argmax(dim=-1)
        ref_correct += reference_outputs.eq(style_labels.to(config.device)).sum().item()

        original_outputs = original_outputs["logits"].argmax(dim=-1)
        original_correct += original_outputs.eq(original_labels.to(config.device)).sum().item()
        
        # for text, predicted_label, true_label in zip(batch['transferred_text'], transferred_outputs, style_labels):
        #     if predicted_label != true_label:
        #         print("transferred:{}\t predicted:{}\t true:{}".format(text, predicted_label, true_label))

    trans_acc = float(trans_correct / len(test_loader.dataset) * 100)
    ref_acc = float(ref_correct / len(test_loader.dataset) * 100)
    original_acc = float(original_correct / len(test_loader.dataset) * 100)
    
    return {"trans_acc": trans_acc, "ref_acc":ref_acc, "original_acc": original_acc}

def evaluate_perplexity(evaluation_files, ppl_model, ppl_tokenizer):

    ppl_model.eval()
    ppl_model.to(config.device)

    test_set = TransferDataset(evaluation_files)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    ppl = 0.

    for batch_idx, batch in enumerate(test_loader):

        transferred_inputs = ppl_tokenizer(batch['transferred_text'], return_tensors="pt")
        transferred_input_ids = torch.tensor(transferred_inputs['input_ids']).long().to(config.device)
    
        original_inputs = ppl_tokenizer(batch["text"], return_tensors="pt")
        original_input_ids = torch.tensor(original_inputs['input_ids']).long().to(config.device)

        with torch.no_grad():
            outputs = ppl_model(transferred_input_ids, labels=original_input_ids)
            ppl += math.exp(outputs.loss)
        
    avg_ppl = ppl / len(test_loader)

    return avg_ppl



if __name__ == "__main__":

    result_files = glob.glob(os.path.join(config.result_path, '*.json'))
    print("evaluation files:{}".format(result_files))

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    # tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=True)
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    #acc 
    pretrained_cls = StyleClassifier.from_pretrained(config.model_dir, style_num=config.num_class)
    
    # #ppl
    # ppl_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    # ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    
    # result_logs_dir = torch.load
    for result_file in result_files:

        model_name = re.findall(r'(.*).json', result_file)
        with open(result_file, 'r') as input_file:
            evaluation_files = json.load(input_file)

        bleu_scores = evaluate_bleu(evaluation_files)
        accs = evaluate_cls(evaluation_files, pretrained_cls, tokenizer, config)
        #ppl = evaluate_perplexity(evaluation_files, ppl_model, ppl_tokenizer)

        print("model name:{}\t bleu:{}\t self bleu:{}\t acc:{}\t".format(model_name, bleu_scores["ref_bleu"], bleu_scores["self_bleu"], accs["trans_acc"]))

