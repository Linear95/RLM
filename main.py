import os
import re
import argparse
import time
import random
import json
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch

import transformers
from transformers import get_cosine_schedule_with_warmup, BertTokenizer

from tools import load_module, save_module
from dataset import TransferDataset, get_dataset_lst
from models import BertForRLM


def prepare_px_training_batch(args, masked_input_ids, origin_input_ids, py_logits, mask_token_id=103):
    batch_size, seq_length = masked_input_ids.shape
    
    masked_indices = (masked_input_ids==mask_token_id).nonzero(as_tuple=True)[1]


    px_input_ids = torch.zeros((batch_size, args.topk, seq_length + 4 + args.topk)).cuda()    # seq + [SEP] + yi + [SEP] + xi + [PAD]
    
    px_labels = []

    for i, mask_idx in enumerate(masked_indices):

        topk_candidates = torch.topk(py_logits[i, mask_idx], args.topk+1).indices.cpu().numpy().tolist()
        yi = np.argmax(py_logits[i, mask_idx].data.cpu().numpy())
        original_token_id = original_input_ids[i, mask_idx]
        xi_candidates = []

        yi_recons_ids = deepcopy(original_input_ids[i])
        yi_recons_ids[mask_idx] = mask_token_id

        if original_token_id not in topk_candidates:
            xi_candidates.append(original_token_id)
            xi_candidates += topk_candidates[1:-1]
        elif original_token_id == yi:
            xi_candidates = topk_candidates[:-1]
        else:
            xi_candidates = topk_candidates[1:]

        random.shuffle(xi_candidates)

        padding = []
        try:
            padding_index = original_input_ids[i].cpu().numpy().tolist().index(0)
            padding = [0 for _ in range(seq_length-padding_index)]
        except:
            padding_index = seq_length

        for candidate_idx, candidate in enumerate(xi_candidates):
            other_candidates = list(set(xi_candidates)-set([candidate]))
            px_input_ids[i, candidate_idx] = torch.tensor([101] + [yi] + [102] + other_candidates + [102] + [candidate] + [102] + yi_recons_ids[1:padding_index].cpu().numpy().tolist() + padding)
        px_labels.append(xi_candidates.index(original_token_id))

        return px_input_ids, px_labels

    

def train(args):
    dataset_lst = get_dataset_lst(data_path=args.train_data_path, mode="pretrain")
    dataset = TransferDataset(dataset_lst)
    
    data_batches = dataset.get_batches(args.batch_size)
    

        
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
    
    model = BertForRLM.from_pretrained(
        args.model_path,
        tokenizer=tokenizer,
        style_num=2,
        markers_path=args.markers_path,
        marker_threshold=args.marker_threshold,
        rarewords_path=args.rarewords_path,
    )

    model.to(args.device)
    
        
    style_optimizer = transformers.AdamW(
        [param for name, param in model.named_parameters() \
         if "bert" in name or "style" in name or "cls" in name or "content_attention" in name],
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    content_optimizer = transformers.AdamW([
        {"params": [param for name, param in model.named_parameters() if "content" in name],\
                                "lr": args.learning_rate},
        {"params": [param for name, param in model.named_parameters() if "bert" in name],\
                                "lr": args.learning_rate},
        ])

    px_optimizer = transformers.AdamW([
        {"params": [param for name, param in model.named_parameters() if "px" in name],\
                                "lr": args.learning_rate},
        {"params": [param for name, param in model.named_parameters() if "bert" in name],\
                                "lr": args.learning_rate / 2},
        ])

    mi_optimizer = torch.optim.Adam(model.mi_estimator.parameters(), lr=args.mi_learning_rate)
    #print([name for name, param in model.mi_estimator.named_parameters()])

    total_steps = len(dataset) * args.training_epochs // args.batch_size

    content_scheduler = get_cosine_schedule_with_warmup(content_optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    style_scheduler = get_cosine_schedule_with_warmup(style_optimizer, num_warmup_steps=0, num_training_steps=total_steps)


    pretrained_epochs = 0
    if args.model_path != 'bert-base-uncased':
        ckpt = load_module(re.sub("\\b" + "model" + "\\b", "optims", args.model_path))
        style_optimizer.load_state_dict(ckpt["style_optim"])
        content_optimizer.load_state_dict(ckpt["content_optim"])
        px_optimizer.load_state_dict(ckpt["px_optim"])
        mi_optimizer.load_state_dict(ckpt["mi_optim"])
        # style_scheduler.load_state_dict(ckpt["style_scheduler"])
        # content_scheduler.load_state_dict(ckpt["content_scheduler"])
        pretrained_epochs = ckpt["epoch"] + 1 

        
    for epoch in range(pretrained_epochs, args.training_epochs):
        
        start_time = time.time()
        
        display_style_loss, display_px_loss, display_content_loss = 0., 0., 0.

        if args.fixed_batch_order:
            shuffled_indices = [i for i in range(len(data_batches))]
            random.shuffle(shuffled_indices)
            data_batches = [data_batches[i] for i in shuffled_indices]

        for batch_idx, batch in enumerate(data_batches):
            #print(batch)
            model.train()            
            style_labels, target_labels, texts, original_texts = batch["label"],\
                            batch["target_label"], batch["text"], batch["original_text"]

            original_tokenized_texts = tokenizer(original_texts, padding=True, return_tensors='pt')
            original_input_ids, original_token_type_ids, original_attention_mask = original_tokenized_texts["input_ids"],\
                 original_tokenized_texts["token_type_ids"], original_tokenized_texts["attention_mask"]

            tokenized_texts = tokenizer(texts, padding=True, return_tensors='pt')
            input_ids, token_type_ids, attention_mask = tokenized_texts["input_ids"],\
                 tokenized_texts["token_type_ids"], tokenized_texts["attention_mask"]

            if input_ids.shape[1] != original_input_ids.shape[1]:
                continue
            batch_size, seq_length = input_ids.shape

            # if seq_length > 50:
            #     continue

            # train MI estimator
            for _ in range(5):
                outputs = model(
                    input_ids=input_ids.to(args.device),
                    token_type_ids=original_token_type_ids.long().to(args.device),
                    attention_mask=original_attention_mask.long().to(args.device),
                    input_styles=torch.tensor(style_labels).long().to(args.device),
                    target_styles=torch.tensor(style_labels).long().to(args.device),
                    contextual_attn_mask=torch.diag(torch.ones(seq_length)).to(args.device),
                    mode="content_training"
                )
                mi_learning_loss = outputs['mi_learning_loss']
                mi_optimizer.zero_grad()
                mi_learning_loss.backward()
                mi_optimizer.step()
                torch.cuda.empty_cache()

            # train content for style disentangling
            content_outputs = model(
                input_ids=input_ids.long().to(args.device),
                token_type_ids=original_token_type_ids.long().to(args.device),
                attention_mask=attention_mask.long().to(args.device),
                input_styles=torch.tensor(style_labels).long().to(args.device),
                target_styles=torch.tensor(style_labels).long().to(args.device),
                target_labels=input_ids.to(args.device),
                mode="content_training",
            )

            content_loss = 0.25 * content_outputs["loss"] + 0.75 * content_outputs["mi_values"].mean() 
            content_optimizer.zero_grad()
            content_loss.backward()
            content_optimizer.step()
        

            # train style classification
            style_outputs = model(
                input_ids=input_ids.long().to(args.device),
                token_type_ids=original_token_type_ids.long().to(args.device),
                attention_mask=attention_mask.long().to(args.device),
                input_styles=torch.tensor(style_labels).long().to(args.device),
                target_styles=torch.tensor(style_labels).long().to(args.device),
                target_labels=original_input_ids.to(args.device),
                mode="style_training",
            )
            style_loss = style_outputs["loss"]

            style_optimizer.zero_grad()
            style_loss.backward()
            style_optimizer.step()

            # if (epoch == 1 and batch_idx >= 10000) or epoch >= 2:
            # if epoch >= 1:
            #     style_scheduler.step()
            #     content_scheduler.step()

            px_loss = 0.
            if epoch >= 4: 

                # train px module
                masked_indices = (input_ids==tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

                with torch.no_grad():
                    py_outputs = model(
                        input_ids=input_ids.long().to(args.device),
                        token_type_ids=original_token_type_ids.long().to(args.device),
                        attention_mask=attention_mask.long().to(args.device),
                        target_styles=torch.tensor(style_labels).long().to(args.device),
                        target_labels=original_input_ids.to(args.device),
                    )

                py_logits = py_outputs["logits"]


                px_input_ids, px_labels = prepare_px_training_batch(args, masked_input_ids, origin_input_ids, py_logits, mask_token_id=tokenizer.mask_token_id)
                
                px_outputs = model(
                    input_ids=px_input_ids.long().to(args.device),
                    masked_recons_ids=input_ids.long().to(args.device),
                    # token_type_ids=original_token_type_ids.long().to(args.device),
                    # attention_mask=attention_mask.long().to(args.device),
                    # target_styles=style_labels.long().to(args.device),
                    target_labels=torch.tensor(px_labels).long().to(args.device),
                    mode="px_training",
                )

                px_optimizer.zero_grad()
                px_loss = px_outputs["loss"]
                px_loss.backward()
                px_optimizer.step()
                torch.cuda.empty_cache()

            display_style_loss += style_loss
            display_content_loss += content_loss
            display_px_loss += px_loss
            
            if batch_idx % 100 == 0 and batch_idx != 0:

                current_time = time.time() - start_time
                model.eval()
                
                print("-----------------------------------------------------------------------------------")
                print("epoch:{}\t batch:{}\t px loss:{}\t style loss:{}\t content loss:{} {}/{}, {}%\t time:{} sec".format(epoch, batch_idx, display_px_loss, 
                    display_style_loss, display_content_loss, batch_idx, len(data_batches), float(batch_idx / len(data_batches)), current_time))

                evaluation_texts = ["it was supposed to be a complete toy .", "so i guess it s really a matter of preference . ", "this product does what it is suppose to do . ", "i had it a long time now and i still love it . "]
                # evaluation_texts = ["the restaurant is very nice.","it 's small yet they make you feel right at home .", "the drinks were affordable and a good pour .", "i will be going back and enjoying this great place!",\
                #              "ever since joes has changed hands it 's just gotten worse and worse .", "so basically tasted watered down .", "there is definitely not enough room in that part of the venue .", "decent selection of meats and cheeses",\
                #                 "nice for me to go and work and have a great breakfast"]

                transferred_texts = model.transfer(evaluation_texts, target_styles=0, device=args.device, topk=args.topk)
                print("************** negative *****************")
                print("transferred:{}".format(transferred_texts))
                # print("candidates:{}".format(candidate_outputs))

                transferred_texts = model.transfer(evaluation_texts, target_styles=1, device=args.device, topk=args.topk)
                print("************** positive *****************")
                print("transferred:{}".format(transferred_texts))
                # print("candidates:{}".format(candidate_outputs))

                display_style_loss = 0.
                display_content_loss = 0.
                display_px_loss = 0.
                
            if batch_idx % 1000 == 0 and batch_idx > 0 and epoch >= 1:
                save_module(model, args.save_dir, module_name='model',
                            additional_name="model_step_{}_{}".format(epoch, batch_idx))
                save_module({"style_optim": style_optimizer, "content_optim": content_optimizer, "mi_optim": mi_optimizer, "px_optim": px_optimizer, "style_scheduler": style_scheduler, "content_scheduler": content_scheduler, "epoch": epoch}, args.save_dir, module_name="optims",\
                                                additional_name="model_step_{}_{}".format(epoch, batch_idx))

def evaluate(args, model, tokenizer):
    test_data = get_dataset_lst(data_path=args.test_data_path, mode="test")
    test_set = TransferDataset(test_data)
    test_data_batches = test_set.get_batches(batch_size=args.test_batch_size)
    
    
    model.to(args.device)
    model.eval()

    output_data = {}
    output_data["text"], output_data["transferred_text"], output_data["reference_text"], output_data["target_label"] = [], [], [], []

    for batch_idx, batch in enumerate(tqdm(test_data_batches)):

        _, target_labels, original_texts, reference_texts = batch["label"], batch["target_label"], batch["text"], batch["reference_text"]

        transferred_texts = model.transfer(
            original_texts, 
            target_styles=target_labels, 
            device=args.device,
            topk=args.topk
        )

        #target_labels = target_labels.cpu().numpy().tolist()

        for text, reference_text, transferred_text, target_label in zip(original_texts, reference_texts, transferred_texts, target_labels):
            output_data["reference_text"].append(reference_text)
            output_data["target_label"].append(target_label)
            output_data["text"].append(text)
            output_data["transferred_text"].append(transferred_text)        

        
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    output_path = os.path.join(args.output_dir, "test_transfer_results.json")
    
    with open(output_path, 'w') as output_file:
        json.dump(output_data, output_file)


def get_args():
    parser = argparse.ArgumentParser("Training Arguments of RLM")

    # env parames
    parser.add_argument("--cuda_devices", type=str, default="0", help="index of gpu devices")

    # data params

    
    parser.add_argument("--model_path", type=str, default='bert-base-uncased', help="model path." )
    parser.add_argument("--tokenizer_path", type=str, default='bert-base-uncased', help='tokenizer path.')
    
    parser.add_argument("--test-output-path", type=str, default="./results_amazon/models_marker_ngrams_lambda25_ordered_new", help="test output path.")
    parser.add_argument("--test_data_path", type=str, default="../data/amazon/test_data.json", help="test data path.")
    parser.add_argument("--test-model-file",type=str, default="model_step_3_10000", help="test model file.")


    parser.add_argument("--save_dir", type=str, default='./saved_models', help='save directory.')
    parser.add_argument("--style_num", type=int, default=2, help="number of styles.")
    parser.add_argument("--topk", type=int, default=3, help="topk candidates in inference.")

    # training params
    parser.add_argument("--train_data_path", type=str, default="../data/masked_train_data.json", help="train data path.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size.")

    parser.add_argument("--training-epochs", type=int, default=3, help="training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="learning rate.")
    parser.add_argument("--mi-learning-rate", type=float, default=1e-4, help="content learning rate.")
    parser.add_argument("--style-learning-rate", type=float, default=5e-5, help="style learning rate.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-8, help="adam optimizer epsilon.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="warmup steps.")
    parser.add_argument("--random-seed", type=int, default=0, help="random seed.")

    # transfer params
    parser.add_argument("--test_batch_size", type=int, default=128, help="test batch size.")
    parser.add_argument("--output_dir", type=str, default='./outputs', help="transferred results output directory")
    
    # marker settings
    parser.add_argument("--markers_path", type=str, default="../data/amazon/markers_ngrams_lambda25.json")
    parser.add_argument("--marker_threshold", type=float, default=1.5)
    parser.add_argument("--rarewords_path", type=str, default="../data/amazon/rare_words_50.json")
    parser.add_argument("--fixed_batch_order", type=bool, default=True)

    parser.add_argument("--evaluate", action='store_true', default=False)


    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


if __name__ == '__main__':
    args = get_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_devices)

    if args.evaluate:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        model = BertForRLM.from_pretrained(
            args.model_path,
            tokenizer=tokenizer,
            style_num=2,
            markers_path=args.markers_path,
            marker_threshold=args.marker_threshold,
            rarewords_path=args.rarewords_path,
        )
        model.to(args.device)
        evaluate(args, model, tokenizer)
    else:
        train(args)    
    

