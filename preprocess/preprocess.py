import os
import torch
import random
import json
import nltk
import argparse
import numpy as np

from copy import deepcopy
from nltk.util import ngrams
from tqdm import tqdm
from transformers import BertTokenizer

def read_file_lines(file_path):
    with open(file_path, "r") as input_file:
        texts_lst = input_file.readlines()
    return texts_lst

def get_dataset(texts, mode=None):

    if mode == "train":
        pos_texts, neg_texts = texts
    elif mode == "test":
        _, _, ref_pos_texts, ref_neg_texts = texts
        pos_texts = [line.strip("\n").split("\t")[0] for line in ref_pos_texts]
        neg_texts = [line.strip("\n").split("\t")[0] for line in ref_neg_texts]
        ref_pos_texts = [line.strip("\n").split("\t")[1] for line in ref_pos_texts]
        ref_neg_texts = [line.strip("\n").split("\t")[1] for line in ref_neg_texts]

    texts = pos_texts + neg_texts
    labels = [1 for _ in range(len(pos_texts))] + [0 for _ in range(len(neg_texts))]
    target_labels = [0 for _ in range(len(pos_texts))] + [1 for _ in range(len(neg_texts))]
    ref_texts = texts if mode == "train" else ref_pos_texts + ref_neg_texts

    labeled_texts = {}
    labeled_texts["text"], labeled_texts["label"], labeled_texts["target_label"], labeled_texts["reference_text"] = [], [], [], []

    for text, label, target_label, ref_text in zip(texts, labels, target_labels, ref_texts):
        text = text.strip()

        labeled_texts["text"].append(text)
        labeled_texts["label"].append(label)
        labeled_texts["target_label"].append(target_label)
        labeled_texts["reference_text"].append(ref_text)
    
    labeled_lst = list(zip(labeled_texts["text"], labeled_texts["label"],\
                                labeled_texts["target_label"], labeled_texts["reference_text"]))
    random.shuffle(labeled_lst)
    labeled_texts["text"], labeled_texts["label"],\
            labeled_texts["target_label"], labeled_texts["reference_text"] = zip(*labeled_lst)

    return labeled_texts

def save_dataset(args, data_set, mode=None):

    output_dict = {}
    if mode == "train":
        for idx, (text, label, target_label) in enumerate(zip(data_set["text"], data_set["label"], data_set["target_label"])):
            output_dict[idx] = {"text":text, "label":label, "target_label":target_label}

    if mode == "test":
        for idx, (text, label, target_label, reference_text) in enumerate(zip(data_set["text"], data_set["label"], data_set["target_label"], data_set["reference_text"])):
            output_dict[idx] = {"text":text, "label":label, "target_label":target_label, "reference_text": reference_text}
    
    output_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filename = os.path.join(output_dir, mode+"_data.json")

    with open(output_filename, "w") as output_file:
        json.dump(output_dict, output_file)

def read_raw_data_files(args):
    data_dir = os.path.join(args.raw_data_dir, args.dataset)
    

    train_pos_file, train_neg_file, test_pos_file, test_neg_file, ref_pos_file, ref_neg_file = "sentiment.train.1", "sentiment.train.0",\
                        "sentiment.test.1", "sentiment.test.0", "reference.1", "reference.0"
    train_pos, train_neg, test_pos, test_neg, ref_pos, ref_neg = [
        read_file_lines(os.path.join(data_dir,file_name))
        for file_name in ["sentiment.train.1", "sentiment.train.0", "sentiment.test.1", "sentiment.test.0", "reference.1", "reference.0"]
    ]
    
    train_set, test_set = get_dataset((train_pos, train_neg), "train"), get_dataset((test_pos, test_neg, ref_pos, ref_neg), "test")
    
    save_dataset(args, train_set, "train")
    save_dataset(args, test_set, "test")

def count_words(texts):

    word_count = {}
    for text in tqdm(texts):
        onegram_in_text = list(ngrams(sequence=nltk.word_tokenize(text), n=1))
        twograms_in_text = list(ngrams(sequence=nltk.word_tokenize(text), n=2))
#         threegrams_in_text = list(ngrams(sequence=nltk.word_tokenize(text), n=3))
#         fourgrams_in_text = list(ngrams(sequence=nltk.word_tokenize(text), n=4))
       
        for token in onegram_in_text:
            if token not in word_count:
                word_count[token] = {}
                word_count[token]["score"] = 0
                word_count[token]["type"] = "1-gram"
            word_count[token]["score"] += 1
        for token in twograms_in_text:
            if token not in word_count:
                word_count[token] = {}
                word_count[token]["score"] = 0
                word_count[token]["type"] = "2-gram"
            word_count[token]["score"] += 1

    return word_count

def compute_markers(stylized_data, opposite_data):
    
    stylization_dict = {}
    
    for item in tqdm(stylized_data.items()):
        key, value = item
        score, ngram_type = value["score"], value["type"]
        
        if key in opposite_data:
            stylization_dict[key] = {}
            stylization_dict[key]["type"] = ngram_type
            stylization_dict[key]["score"] = float((value["score"]+ args.alpha)/(opposite_data[key]["score"]+args.alpha))

    return stylization_dict

def compute_ngram_score(stylization_dict, onegram, twograms):
    
    onegram_score = stylization_dict[onegram]
    if len(twograms)!= 0:
        twogram_score = sum([stylization_dict[twogram] for twogram in twograms]) / len(twograms)
        token_score = 0.25 *onegram_score + 0.75 * twogram_score 
    else:
        token_score = onegram_score
        
    return token_score

def get_markers_and_rarewords(args):

    # get train file
    train_dir = os.path.join(args.data_dir, args.dataset)

    with open(os.path.join(train_dir, "train_data.json"), "r") as input_file:
        data = json.load(input_file)
    
    pos_data, neg_data, all_data = [], [], []
    for item in data.items():
        key, value = item
        text, label = value["text"], value["label"]
        if label == 1:
            pos_data.append(text)
        else:
            neg_data.append(text)
            
        all_data.append(text)
    
    pos_stat, neg_stat, all_stat = count_words(pos_data), count_words(neg_data), count_words(all_data)
    pos_markers, neg_markers = compute_markers(pos_stat, neg_stat), compute_markers(neg_stat, pos_stat)

    # compute markers
    output_markers = {}
    for item in pos_markers.items():
        key, value = item
        pos_value = value["score"]
        neg_value = neg_markers[key]["score"]
        marker_value = max(pos_value, neg_value)
        output_markers[" ".join(key)] = marker_value

    # compute rare words
    rare_words = []
    for item in all_stat.items():
        key, value = item
        ngram_type, score = value["type"], value["score"]
        if ngram_type == "1-gram" and score <= args.beta:
            rare_words.append(key[0])
    
    output_dir = os.path.join(args.data_dir, args.dataset)
    markers_output_filename = os.path.join(output_dir, "markers.json")
    with open(markers_output_filename, "w") as output_file:
        json.dump(output_markers, output_file)

    rarewords_output_filename = os.path.join(output_dir, "rare_words.json")
    with open(rarewords_output_filename, "w") as output_file:
        json.dump(rare_words, output_file)


def get_masked_train_data(args):

    data_dir = os.path.join(args.data_dir, args.dataset)

    # stopwords
    # with open(os.path.join(data_dir, "stopwords_tokenized.json"), "r") as input_file:
    #     stopwords = json.load(input_file)
    stopwords = []
    
    # train data
    with open(os.path.join(data_dir, "train_data.json"), "r") as input_file:
        train_dict = json.load(input_file)
    # rare words
    with open(os.path.join(data_dir, "rare_words.json"), "r") as input_file:
        rare_words = json.load(input_file)
    # markers
    with open(os.path.join(data_dir, "markers.json"), "r") as input_file:
        all_markers = json.load(input_file)
    
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    masked_train_dict = {}

    for item in tqdm(train_dict.items()):
        
        key, value = item
        text, label, target_label = value["text"], value["label"], value["target_label"]

        tokenized_text = tokenizer.tokenize(text)
        
        drop_flag = False

        if len(tokenized_text) < 5:
            continue
        patience = 0
        
        subword_ids, number_ids, rareword_ids = [], [], []
        
        for token_idx, token in enumerate(tokenized_text):
            if "##" in token or (token_idx+1 < len(tokenized_text) and "##" in tokenized_text[token_idx+1]):
                subword_ids.append(token_idx)
            if token_idx+1 < len(tokenized_text) and token == "_" and tokenized_text[token_idx+1] == "extend":
                number_ids.append(token_idx+1)
            if token in rare_words:
                rareword_ids.append(token_idx)
        
        masked_token = np.random.choice(len(tokenized_text), 1)[0]
        
        masked_tokenized_text = deepcopy(tokenized_text)
        #print('debug 1', masked_tokenized_text)
        
        while True:
            
            while (masked_token in subword_ids) or (masked_token in number_ids) or (masked_token in rareword_ids) or (tokenized_text[masked_token] in stopwords) or (tokenized_text[masked_token].isalpha()==False):
                masked_token = np.random.choice(len(tokenized_text), 1)[0]
                patience += 1
                if patience >= 20:
                    break
                continue

            previous_token = None if masked_token-1 <0 or tokenized_text[masked_token-1].isalpha()==False else tokenized_text[masked_token-1]
            next_token = None if masked_token+1>= len(tokenized_text) or tokenized_text[masked_token+1].isalpha()==False else tokenized_text[masked_token+1]
            twograms = []
        
            if previous_token is not None:
                twogram_1 = " ".join((previous_token, tokenized_text[masked_token]))
                if twogram_1 in all_markers:
                    twograms.append(twogram_1)
            if next_token is not None:
                twogram_2 = " ".join((tokenized_text[masked_token], next_token))
                if twogram_2 in all_markers:
                    twograms.append(twogram_2)

            if tokenized_text[masked_token] not in all_markers:
                ngram_score = 0
            else:
                ngram_score = compute_ngram_score(all_markers, tokenized_text[masked_token], twograms)
            
            if ngram_score >= args.marker_threshold:
                masked_tokenized_text[masked_token] = "[MASK]"
                break
            else:  
                patience += 1
                if patience >= 20:
                    drop_flag = True
                    patience = 0
                    break

        if drop_flag == True:
            continue
        
        masked_tokenized_text = tokenizer.decode(tokenizer.encode(masked_tokenized_text)).strip("[CLS]").strip("[SEP]")
        #print('debug 2', masked_tokenized_text)
        tokenized_text = tokenizer.decode(tokenizer.encode(tokenized_text)).strip("[CLS]").strip("[SEP]")

        masked_train_dict[key] = {"text":masked_tokenized_text, "original_text": tokenized_text, "label":label, "target_label": target_label}
    
    masked_train_dict_ordered = {}
    masked_train_dict_ordered_tuples = sorted(masked_train_dict.items(), key=lambda x: len(tokenizer.tokenize(x[1]["text"])))
    for pair_idx, pair in enumerate(masked_train_dict_ordered_tuples):
        key, value = pair
        #print(value)
        masked_train_dict_ordered[pair_idx] = value

        
    output_dir = os.path.join(args.data_dir, args.dataset)
    masked_output_filename = os.path.join(output_dir, "masked_train_data.json")

    with open(masked_output_filename, "w") as output_file:
        json.dump(masked_train_dict_ordered, output_file)


def get_args():

    parser = argparse.ArgumentParser("replace-language-model data preprocessing")

    # env params
    parser.add_argument("--cuda_devices", type=str, default="0", help="index of gpu devices")

    parser.add_argument("--data_dir", type=str, default="../data_test", help="data directory.")
    parser.add_argument("--raw_data_dir", type=str, default="", help="raw data directory.")
    parser.add_argument("--dataset", type=str, default="yelp", help="data source.")
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-uncased", help="tokenizer path")

    # marker params
    parser.add_argument("--alpha", type=int, default=25, help="smoothing parameter for marker computation.")
    parser.add_argument("--beta", type=int, default=25, help="rare words threshold.")
    parser.add_argument("--marker_threshold", type=float, default=1.5, help="marker threshold.")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args

if __name__ == "__main__":

    args = get_args()

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    print("reading raw data files ...")
    read_raw_data_files(args)
    
    print("get markers and rarewords ...")
    get_markers_and_rarewords(args)
    
    print("get masked train data ...")        
    get_masked_train_data(args)
    print("preprocessing finished.")

