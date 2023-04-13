import re
import argparse
import time
import random

import json
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from mi_estimators import CLUBForCategorical

class BertForRLM(BertPreTrainedModel):

    def __init__(self, config, tokenizer, style_num, markers_path, marker_threshold, rarewords_path):
        '''
        num_seq_labels & num_token_labels : [head1_label_num, head2_label_num, ..., headn_label_num]
        '''
        super().__init__(config)
        #print(config)

        self.style_num = style_num
        self.vocab_size = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=True)
        self.cls = BertOnlyMLMHead(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        # style
        self.style_embeddings = nn.Embedding(num_embeddings=style_num, embedding_dim=config.hidden_size)
        self.style_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.style_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.style_concat_transform = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.style_attention = nn.MultiheadAttention(embed_dim=config.hidden_size * 2,  kdim=config.hidden_size * 2, vdim=config.hidden_size * 2, num_heads=1, batch_first=True)  


        # content
        self.content_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.content_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.content_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.content_attention = nn.MultiheadAttention(embed_dim=config.hidden_size,  kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)        
        
        # px
        self.px_pooler = self.bert.pooler
        self.px_transform = nn.Linear(config.hidden_size, 1)
        self.tokenizer = tokenizer

        self.stopwords = ["why", "these", "there", "their", "whom", "that", "how", "here", "his", "herself", "your",\
             "my", "hers", "we", "she", "myself", "they", "them", "her", "me", "yourselves", "this",  "ours", "its",\
                 "it", "ourselves", "which", "yours", "themselves", "where", "itself", "those", "what", "when", "our",\
                     "you",  "who", "theirs", "he", "him", "yourself", "himself", "i", "a", "an", "the"]

        self.special_token_ids = self.tokenizer.encode(".[PAD]") + self.tokenizer.convert_tokens_to_ids(self.stopwords)

        self.mi_estimator = CLUBForCategorical(input_dim=config.hidden_size, label_num=style_num)

        self.markers_path = markers_path
        self.marker_threshold = marker_threshold
        
        with open(self.markers_path, "r") as input_file:
            markers = json.load(input_file)
        self.markers = markers

        with open(rarewords_path, "r") as input_file:
            rare_words = json.load(input_file)
        self.rare_words = rare_words


    def initialize_style_embeddings(self, style_weights):
        self.style_embeddings.from_pretrained(style_weights)


    def forward(
            self,
            input_ids=None,
            masked_recons_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_styles=None,
            target_styles=None,
            target_labels=None,
            contextual_attn_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            mask_token_id=103,
            mask_idx=None,
            mode=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if len(input_ids.shape) == 3:
            num_choices = input_ids.shape[1]
            input_ids = input_ids.view(-1, input_ids.size(-1))
            
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs['last_hidden_state']
        batch_size, seq_length, embed_dim = hidden_state.shape

        masked_token_indices = (input_ids==mask_token_id).nonzero(as_tuple=True)
        masked_cols, masked_elems = masked_token_indices[0].cuda(), masked_token_indices[1].cuda()

        padded_token_indices = (input_ids==0).nonzero(as_tuple=True)
        padded_cols, padded_elems = padded_token_indices[0].cuda(), padded_token_indices[1].cuda()

        if mask_idx is None:
            # context tokens: predict from only context tokens
            content_attn_mask = torch.diag(torch.ones(seq_length)).bool().cuda().unsqueeze(0).repeat(batch_size, 1, 1)
            content_attn_mask[padded_cols, padded_elems, :] = True
            content_attn_mask[padded_cols, :, padded_elems] = False

            if mode == "content_training":
                content_attn_mask[masked_cols, masked_elems, :] = True
                content_attn_mask[masked_cols, :, masked_elems] = False
        else:
            content_attn_mask = torch.diag(torch.ones(seq_length)).bool().cuda()

        # content_attn_mask = torch.ones(size=(seq_length, seq_length)).cuda() - content_attn_mask
        # content_attn_mask = content_attn_mask.bool()
        # content_attn_mask[masked_elems, :] = False

        style_attn_mask = torch.diag(torch.ones(seq_length)).cuda()
        style_attn_mask = torch.ones(size=(seq_length, seq_length)).cuda() - style_attn_mask
        style_attn_mask = style_attn_mask.bool()   


        # capture content information
        content_emb = self.content_transform(hidden_state)
        content_outputs = self.content_layernorm(content_emb)
        content_attn_outputs = self.content_attention(query=content_outputs, key=content_outputs, value=content_outputs, attn_mask=content_attn_mask, need_weights=False)[0]
        content_attn_outputs = self.content_attention_layernorm(self.dropout(content_attn_outputs)+content_outputs)


        if target_styles is not None:

            # capture style information
            target_style_emb = self.style_embeddings(target_styles)
            target_style_emb = target_style_emb.unsqueeze(1).repeat(1, seq_length, 1)
            concat_emb = torch.concat((content_attn_outputs, target_style_emb), -1)
            style_emb = self.style_attention(query=concat_emb, key=concat_emb, value=concat_emb, attn_mask=style_attn_mask, need_weights=False)[0]
            style_outputs = self.style_concat_transform(style_emb)
            style_outputs = self.style_transform(style_outputs)
            # output_emb = self.style_layernorm(style_outputs)
            output_emb = self.style_layernorm(self.dropout(style_outputs)+content_outputs)

            #generate
            logits = self.cls(output_emb)

        else:

            # masked_token_indices = (masked_recons_ids.view(-1, masked_recons_ids.size(-1))==tokenizer.mask_token_id).nonzero(as_tuple=True)
            # masked_cols, masked_elems = masked_token_indices[0].cuda(), masked_token_indices[1].cuda()

            px_pooled_outputs = self.px_pooler(hidden_state)
            px_outputs = self.dropout(px_pooled_outputs)
            logits = self.px_transform(px_outputs)

            logits = logits.view(-1, num_choices)

        
        loss = None
        if target_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss

            if mode == "style_training":
                active_loss = input_ids.view(-1) == self.tokenizer.mask_token_id
                active_labels = torch.where(
                    active_loss, target_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(target_labels)
                )
                active_logits = logits.view(-1, self.vocab_size)
            elif mode == "px_training":
                active_logits = logits
                active_labels = target_labels
            else:
                # active_loss = input_ids.view(-1) != 0
                # active_labels = torch.where(
                #     active_loss, target_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(target_labels)
                # )
                active_labels = target_labels.view(-1)
                active_logits = logits.view(-1, self.vocab_size)

            # else:
            #     active_labels = target_labels.view(-1)
            
            loss = loss_fct(active_logits, active_labels)

        mi_learning_loss = None
        mi_values = None
        if input_styles is not None:
            input_styles_extend = input_styles.unsqueeze(1).repeat(1, seq_length) #[batch_size, seq_length]
            
            mi_learning_loss = self.mi_estimator.learning_loss(
                inputs=content_outputs.reshape(-1, embed_dim).detach(),
                labels=input_styles_extend.reshape(-1)
            )

            mi_values = torch.stack([self.mi_estimator(
                inputs=content_outputs[:,token_idx,:],
                labels=input_styles,
                ) for token_idx in range(seq_length)
            ])

        return {
            "logits": logits,
            "loss": loss,
            "mi_learning_loss": mi_learning_loss,
            "mi_values": mi_values,
            }

    def get_reserved_ids(self, input_ids, generate_subword=False):
        reserved_ids = []            
        for input_token_ids in input_ids:
            reserved_token_ids = []
            for token_idx, input_token_id in enumerate(input_token_ids):
                if input_token_id in self.special_token_ids:
                    reserved_token_ids.append(1)
                    continue

                if not generate_subword:
                    if "##" in self.tokenizer.decode([input_token_id]):
                        reserved_token_ids.append(1)
                        continue
                    if  token_idx < len(input_ids) -1 and '##' in self.tokenizer.decode([input_token_ids[token_idx + 1]]):
                        reserved_token_ids.append(1)
                        continue
                reserved_token_ids.append(0)
            reserved_ids.append(reserved_token_ids)
        #print(reserved_ids)
        return reserved_ids
    
    def compute_ngram_score(self, stylization_dict, onegram, twograms):
        type2coeff = {"1-gram":0.25, "2-gram": 0.75}
        try:
            onegram_score = stylization_dict[onegram]
            if len(twograms)!= 0:
                # twogram_score = sum([stylization_dict[twogram] for twogram in twograms]) / len(twograms)
                token_score = sum([stylization_dict[twogram] for twogram in twograms]) / len(twograms) 
                # token_score = 0.25*onegram_score + 0.75 * twogram_score 
            else:
                token_score = onegram_score
        except:
            token_score = 0
            
        return token_score

    def transfer(self, inputs, target_styles, device, topk=3):

        num_samples = len(inputs)
        if isinstance(target_styles, int):
            target_styles = torch.tensor(target_styles).repeat(num_samples)
        else:
            target_styles = torch.tensor(target_styles)

        reversed_styles = torch.abs(target_styles - torch.ones(target_styles.shape))

        output_texts = []

        candidate_outputs = []

        for sample_idx in range(num_samples):

            output_tokens = []

            sample_markers = []
            split_sample = inputs[sample_idx].split()
            
            for token_idx, token in enumerate(split_sample):
                previous_token = None if token_idx-1 <0 or split_sample[token_idx-1].isalpha()==False else split_sample[token_idx-1]
                next_token = None if token_idx+1 >= len(split_sample) or split_sample[token_idx+1].isalpha()==False else split_sample[token_idx+1]
                twograms = []
                if previous_token is not None:
                    twogram_1 = " ".join((previous_token, token))
                    if twogram_1 in self.markers:
                        twograms.append(twogram_1)
                if next_token is not None:
                    twogram_2 = " ".join((token, next_token))
                    if twogram_2 in self.markers:
                        twograms.append(twogram_2)
                    
                if token not in self.markers:
                    ngram_score = 0
                else:
                    ngram_score = self.compute_ngram_score(self.markers, token, twograms)
                    
                if ngram_score <= self.marker_threshold:
                    sample_markers += self.tokenizer.tokenize(token)
                
            tokenized_inputs = self.tokenizer(inputs[sample_idx], return_tensors="pt")
            input_ids, attention_mask, token_type_ids = tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], tokenized_inputs["token_type_ids"]
            original_input_ids = deepcopy(input_ids)

            seq_length = input_ids.shape[1]

            target_style, reversed_style = target_styles[sample_idx], reversed_styles[sample_idx]

            original_input_ids = deepcopy(input_ids)

            ##test
            sample_candidate_tuples = []
            
            for mask_idx in range(seq_length):

                original_token_id = deepcopy(input_ids[0, mask_idx])
                original_token = "".join(self.tokenizer.decode(original_token_id).split())

                if mask_idx <= seq_length -2:
                    next_token = "".join(self.tokenizer.decode(input_ids[0, mask_idx+1]).split())
                else:
                    next_token = original_token
                
                if "##" in next_token or "##" in original_token:
                    output_tokens.append(original_token_id)
                    continue

                if original_token_id in [101, 102]:
                    output_tokens.append(original_token_id)
                    continue
            
                if original_token in self.stopwords:
                    output_tokens.append(original_token_id)
                    continue

                if original_token.isalpha()==False and original_token != "'":
                    output_tokens.append(original_token_id)
                    continue
                    
                if original_token in sample_markers:
                    output_tokens.append(original_token_id)
                    continue

                if original_token in self.rare_words:
                    output_tokens.append(original_token_id)
                    continue


                input_ids[0, mask_idx] = 103
                
                candidate_tuples = []

                with torch.no_grad():
                    outputs = self.forward(
                        input_ids=input_ids.long().to(device),
                        attention_mask=attention_mask.long().to(device),
                        token_type_ids=token_type_ids.long().to(device),
                        target_styles=target_style.unsqueeze(0).long().to(device),
                        mask_idx=mask_idx,
                        mode=""
                        )

                logits = torch.log(F.softmax(outputs["logits"], dim=-1))
                #style_token_id = logits[0, mask_idx].argmax(axis=-1)

                #candidates = torch.topk(logits[0, mask_idx], config.topk).indices.cpu().numpy().tolist()
                candidates = torch.topk(logits[0, mask_idx], 1).indices.cpu().numpy().tolist()
                candidates.append(original_token_id)

                topk_candidates = torch.topk(logits[0, mask_idx], topk+1).indices.cpu().numpy().tolist()

                xi_candidates = []
                if original_token_id not in topk_candidates:
                    xi_candidates.append(original_token_id.cpu().numpy().tolist())
                    xi_candidates += topk_candidates[1:-1]
                elif original_token_id == np.argmax(logits[0, mask_idx].data.cpu().numpy()):
                    xi_candidates = topk_candidates[:-1]
                else:
                    xi_candidates = topk_candidates[1:]
                    # print("original token id:", original_token_id)
                    # print("original token idx:", xi_candidates.index(original_token_id.cpu().numpy().tolist()))
                    # print("xi candidates", xi_candidates)

                original_token_idx = xi_candidates.index(original_token_id.cpu().numpy().tolist())

                loglikeli_candidates = []

                ##test
                candidate_tuples = []

                for candidate in candidates:

                    loglikeli_y = torch.tensor(logits[0, mask_idx, candidate]).cpu().numpy()
                    
                    px_input_ids = torch.zeros((1, topk, seq_length + 4 + topk)).cuda()

                    yi_recons_ids = deepcopy(original_input_ids)
                    yi_recons_ids[0, mask_idx] = self.tokenizer.mask_token_id

                    for candidate_idx, xi_candidate in enumerate(xi_candidates):
                        other_candidates = list(set(xi_candidates)-set([xi_candidate]))
                        # print("xi candidates:{}\t xi candidate:{}\t other candidates:{}".format(xi_candidates, xi_candidate, other_candidates))
                        px_input_ids[0, candidate_idx] = torch.tensor([[101] + [candidate] + [102] + other_candidates + [102] + [xi_candidate] + [102] + yi_recons_ids[0, 1:].cpu().numpy().tolist() ])

                    with torch.no_grad():
                        candidate_outputs = self.forward(
                            input_ids=px_input_ids.long().to(device),
                            masked_recons_ids=input_ids.long().to(device),
                            mask_idx=mask_idx,
                            # attention_mask=attention_mask.long().to(device),
                            # token_type_ids=token_type_ids.long().to(device),
                            # target_styles=target_style.unsqueeze(0).long().to(device),
                            # mask_idx=mask_idx
                            )

                    #outputs["logits"][0, mask_idx, candidate] = 1e-3
                    candidate_logits = torch.log(F.softmax(candidate_outputs["logits"], dim=-1))
                    loglikeli_x = torch.tensor(candidate_logits[0, original_token_idx]).cpu().numpy()

                    # loglikeli_x = torch.log(candidates_outputs["logits"][0][0]).cpu().numpy()#+0.03)
                    # loglikeli_x = torch.tensor(candidates_logits).cpu().numpy()
                    loglikeli_candidate = 0.0 + loglikeli_y # loglikeli_x + loglikeli_y
                    loglikeli_candidates.append(loglikeli_candidate)

                    # token_candidate = "".join(self.tokenizer.decode(candidate).split())

                    # loglikeli_x_max = torch.max(candidates_logits[0, mask_idx]).cpu().numpy().tolist()
                    # loglikeli_x_argmax = candidates_logits[0, mask_idx].argmax(axis=-1).cpu().numpy().tolist()
                    # loglikeli_candidate = "".join(self.tokenizer.decode(loglikeli_x_argmax).split())
                    # print(original_token, token_candidate, loglikeli_candidate, loglikeli_x.tolist(), loglikeli_y.tolist(), loglikeli_x_max, loglikeli_x_argmax, loglikeli_candidate)

                # sample_candidate_tuples.append(candidate_tuples)
 
                loglikeli_candidates = torch.tensor(loglikeli_candidates)
                predicted_token_idx = loglikeli_candidates.argmax(axis=-1)

                predicted_token_id = candidates[predicted_token_idx]
                
                # destylized_prob_original_token = torch.tensor(destylized_logits[0, mask_idx, original_token_id]).detach().cpu().numpy()
                # original_difference_output = style_prob_original_token - destylized_prob_original_token

                # destylized_prob_style_token = torch.tensor(destylized_logits[0, mask_idx, style_token_id].detach().cpu().numpy())
                # transferred_difference_output = style_prob_style_token - destylized_prob_style_token

                #predicted_token_id = style_token_id if transferred_difference_output > original_difference_output else original_token_id

                input_ids[0, mask_idx] = predicted_token_id
                
                output_tokens.append(predicted_token_id)

            
            output_text = self.tokenizer.decode(output_tokens)
            output_texts.append(output_text)

            # candidate_outputs.append(sample_candidate_tuples)

        return output_texts
 
    def transfer_batch(self, sentences, target_styles, topk=3, generate_subword=False):

        if isinstance(sentences, str):
            sentences = [sentences]

        batch_size = len(sentences)
        
        if isinstance(target_styles, int):
            target_styles = [target_styles] * batch_size


        tokenized_inputs = self.tokenizer(sentences, padding=True)
        input_ids, attention_mask, token_type_ids = tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], tokenized_inputs["token_type_ids"]
        reserved_ids = self.get_reserved_ids(input_ids, generate_subword=generate_subword)
        #print(reserved_ids)


        output_ids = [[self.tokenizer.cls_token_id] + [self.tokenizer.pad_token_id]*(len(input_ids[0])-1) for k in range(batch_size)] 
        #print(output_ids)
        for mask_idx in range(1,len(input_ids[0])-1):

            combined_ids =[output_ids[k][:mask_idx] + [self.tokenizer.mask_token_id] + input_ids[k][mask_idx+1:]
                             for k in range(batch_size)]
            #print(combined_ids)
            
            mask_col = [i for i in range(batch_size)]
            seq_length = len(input_ids[0])

            # context tokens: predict from only masked tokens
#             style_attn_mask = np.zeros((batch_size, seq_length, seq_length))
#             style_attn_mask[mask_col, :, mask_idx] = 1.

#             print("combined shape:{}\t mask shape:{}".format(torch.tensor(combined_ids).shape, torch.tensor(style_attn_mask).shape))
            # # # masked token: predict from only context tokens
            # style_attn_mask[mask_col, mask_idx, :] = 1.
            # style_attn_mask[mask_col, mask_idx, mask_idx] = 0.

            with torch.no_grad():
                outputs = self.forward(
                    input_ids=torch.tensor(combined_ids).long().to(self.device),
                    attention_mask=torch.tensor(attention_mask).long().to(self.device),
                    token_type_ids=torch.tensor(token_type_ids).long().to(self.device),
                    target_styles=torch.tensor(target_styles).long().to(self.device),
                    mask_token_id=self.tokenizer.mask_token_id,
                    mask_idx=mask_idx,
                    mode="generate"
                    )

            likelihood_y = F.softmax(outputs["logits"], dim=-1)[:, mask_idx]  #[batch_size, vocab_size]
            #style_token_id = logits[0, mask_idx].argmax(axis=-1)


            candidate_ids = torch.topk(likelihood_y, topk, dim=-1).indices.cpu().numpy().tolist()
            candidate_ids = [candidate_ids[k] + [input_ids[k][mask_idx]] for k in range(batch_size)]
            #print(candidate_ids)

            ##test
            #rank_y = {candidate_id: i for i, candidate_id in enumerate(candidate_ids)}

            predict_id = []
            max_likelihood = 0.

            max_scores = [0.] * batch_size
            pred_ids = [0] * batch_size
            for c_i in range(topk+1):

                #rec_input_ids = [[candidate_ids[k][c_i]] + output_ids[k][1: mask_idx] + [self.tokenizer.mask_token_id] + input_ids[k][mask_idx+1:] for k in range(batch_size)]
                rec_input_ids = [input_ids[k][: mask_idx] + [candidate_ids[k][c_i]] + input_ids[k][mask_idx+1:] for k in range(batch_size)]
                #print(rec_input_ids)
                #print(input_ids)
                with torch.no_grad():
                    candidate_outputs = self.forward(
                        input_ids=torch.tensor(rec_input_ids).long().to(self.device),
                        attention_mask=torch.tensor(attention_mask).long().to(self.device),
                        target_styles=torch.tensor(target_styles).long().to(self.device),
                        token_type_ids=torch.tensor(token_type_ids).long().to(self.device),
                        mask_token_id=self.tokenizer.mask_token_id
                        )

                #outputs["logits"][0, mask_idx, candidate] = 1e-3

                likelihood_x = F.softmax(candidate_outputs["logits"], dim=-1)[:, mask_idx] #[batchsize,vocabsize]                    

                p_scores = [(likelihood_y[k, candidate_ids[k][c_i]] * likelihood_x[k, input_ids[k][mask_idx]]).item() for k in range(batch_size)]

                for k, p_score in enumerate(p_scores):
                    if p_score > max_scores[k]:
                        max_scores[k] = p_score
                        pred_ids[k] = candidate_ids[k][c_i]
                        
                #print(pred_ids)


            for k in range(batch_size):
                output_ids[k][mask_idx] = pred_ids[k] if reserved_ids[k][mask_idx] == 0 else input_ids[k][mask_idx]

            # print(output_ids)
            # min_rank = topk * 3
            # for candidate_id in candidate_ids:
            #     if rank_x[candidate_id] + rank_y[candidate_id] < min_rank:
            #         min_rank = rank_x[candidate_id] + rank_y[candidate_id]
            #         predict_id = candidate_id

            #print(output_ids)



        # transferred_sent = self.tokenizer.decode(output_ids,skip_special_tokens=True)
        # transferred_sentences.append(transferred_sent)
        transfer_results = [self.tokenizer.decode(output_token_ids,skip_special_tokens=True) for output_token_ids in output_ids]

            #candidate_outputs.append(sample_candidate_tuples)

        return transfer_results
