
import warnings

warnings.simplefilter(action="ignore", category=(FutureWarning, DeprecationWarning))

import json
import numpy as np
import torch
from tqdm.auto import tqdm
from ufal.chu_liu_edmonds import chu_liu_edmonds
from probing.utilities import *


class PerturbedProbe(object):
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.layers = int(self.args.layers) + 1

        if self.args.cuda:
            self.model.to('cuda')

        self.model = self.model.eval()

    def perturbe(self, sentence, label, root_id, ud_deprels, results):
        # generate masks
        # get id for MASK
        mask_id = self.tokenizer.mask_token_id
        
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.encode(sentence)
        tokenized_text = self.tokenizer.convert_ids_to_tokens(indexed_tokens)

        # additional check for unk tokens
        if '[UNK]' in tokenized_text or '<unk>' in tokenized_text:
            return results
        
        ud_sent = sentence.split()
        
        # map tokens to words
        if self.args.model in ["bert-base-multilingual-cased"]:
            mapping = map_seq_bert(tokenized_text, ud_sent)
        else:
            mapping = map_seq_xlmr(tokenized_text, ud_sent)
        
        # generating mask indices
        all_layers_matrix_as_list = [[] for i in range(self.layers)]

        len_tokens = len(tokenized_text)
        
        for i in range(0, len_tokens):
            
            id_for_all_i_tokens = get_subwords(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            
            # mask all bpe tokens of a word
            for tmp_id in id_for_all_i_tokens:
                tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(len_tokens)]
            
            for j in range(len_tokens):
                id_for_all_j_tokens = get_subwords(mapping, j)
                # mask all bpe tokens of a word
                for tmp_id in id_for_all_j_tokens:
                    one_batch[j][tmp_id] = mask_id 
    
            tokens_tensor = torch.tensor(one_batch)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
            if self.args.cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')
            
            # get hidden states
            with torch.no_grad():
                if self.args.model in ["facebook/mbart-large-cc25"]:
                    model_outputs = self.model(tokens_tensor)
                    all_layers = model_outputs.encoder_hidden_states  # 12 layers + embedding layer
                elif self.args.model in ['xlm-roberta-base']:
                    model_outputs = self.model(tokens_tensor)
                    all_layers = model_outputs.hidden_states  # 12 layers + embedding layer
                elif self.args.model = ['bert-base-multilingual-cased']:
                    model_outputs = self.model(tokens_tensor, segments_tensor)
                    all_layers = model_outputs.hidden_states  # 12 layers + embedding layer

            # get hidden states for word_i
            for k, layer in enumerate(all_layers):
                if self.args.cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
                    hidden_states_for_token_i = layer[:, i, :].numpy()
                all_layers_matrix_as_list[k].append(hidden_states_for_token_i)
                
            del one_batch, tokens_tensor, segments_tensor, model_outputs
            
        for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
            init_matrix = np.zeros((len_tokens, len_tokens))

            for i, hidden_states in enumerate(one_layer_matrix):
                base_state = hidden_states[i]

                for j, state in enumerate(hidden_states):
                    if self.args.metric == 'dist':
                        init_matrix[i][j] = np.linalg.norm(base_state - state)
                    if self.args.metric == 'cos':
                        init_matrix[i][j] = np.dot(base_state, state) / (
                                np.linalg.norm(base_state) * np.linalg.norm(state))
                                
            results[k].append((sentence, label, tokenized_text, root_id, ud_deprels, ud_sent, init_matrix, mapping))
        
        return results
        
    def extract_matrices(self):
        
        results = [[] for _ in range(self.layers)]

        for (label, correct_sent, incorrect_sent, cor_root, inc_root, ud_deprels) in tqdm(self.data, total=len(self.data)):
         
            results = self.perturbe(correct_sent, 'O', cor_root, ud_deprels, results)
            results = self.perturbe(incorrect_sent, 'I', inc_root, ud_deprels, results)

        return results

    def extract_trees(self, results):
        
        trees = {
            layer: [] for layer in range(self.args.layers + 1)
        }

        for layer, state in tqdm(enumerate(results), total=len(results)):

            for s, (sentence, label, tokens, root_id, ud_deprels, ud_sent, sent_matrix, mapping) in enumerate(state):
        
                init_matrix = sent_matrix

                # merge subwords in one row
                merge_column_matrix = []
                for i, line in enumerate(init_matrix):
                    new_row = []
                    buf = []
                    for j in range(0, len(line)):
                        buf.append(line[j])
                        if j != len(mapping) - 1:
                            if mapping[j] != mapping[j + 1]:
                                new_row.append(buf[0])
                        else:
                            new_row.append(buf[0])
                        buf = []
                    merge_column_matrix.append(new_row)
               
                # merge subwords in multi rows
                # transpose the matrix so we can work with row instead of multiple rows
                merge_column_matrix = np.array(merge_column_matrix).transpose()
                merge_column_matrix = merge_column_matrix.tolist()
                final_matrix = []
                for i, line in enumerate(merge_column_matrix):
                    new_row = []
                    buf = []
                    for j in range(0, len(line)):
                        buf.append(line[j])
                        if j != len(mapping) - 1:
                            if mapping[j] != mapping[j + 1]:
                                if self.args.subword == 'sum':
                                    new_row.append(sum(buf))
                                elif self.args.subword == 'avg':
                                    new_row.append((sum(buf) / len(buf)))
                                elif self.args.subword == 'first':
                                    new_row.append(buf[0])
                        else:
                            new_row.append(buf[0])
                        buf = []
                    final_matrix.append(new_row)

                # transpose to the original matrix
                final_matrix = np.array(final_matrix).transpose()
                if self.args.model in ["facebook/mbart-large-cc25"]:
                    final_matrix = final_matrix[:-1, :-1]
                else:
                    final_matrix = final_matrix[1:-1, 1:-1]
                
                # put gold root on 0 position
                root = final_matrix[root_id].copy()
                final_matrix[root_id], final_matrix[0] = final_matrix[0], root

                if self.args.model in ["facebook/mbart-large-cc25"]:
                    perturbed_sent = merge_tokens(tokens, mapping)[:-1]
                elif self.args.model in ["bert-base-multilingual-cased"]:
                    perturbed_sent = merge_tokens(tokens, mapping)[1:-1]
                elif self.args.model in ["xlm-roberta-base"]:
                    perturbed_sent = merge_tokens(tokens, mapping)[1:-1]
                    
                perturbed_sent[root_id], perturbed_sent[0] = perturbed_sent[0], perturbed_sent[root_id]

                # extract deprels
                heads, _ = chu_liu_edmonds(final_matrix)
                deprels = get_deprels(heads, perturbed_sent)
                
                assert len(deprels) == len(ud_deprels)
                    
                trees[layer].append((label, deprels, ud_deprels))
             
        return trees
        
    def evaluate(self, probe_data):
            
        results = {'layers': {layer: {} for layer in range(len(probe_data))}}
        
        for layer, trees in tqdm(probe_data.items(), total=len(probe_data), desc='Layer-wise evaluation'):
            
            layer_results = {
                'O': {'uas': [], 'uuas': []}, 
                'I': {'uas': [], 'uuas': []},
            }
            
            for ind, (label, tree, ud_tree) in enumerate(trees):
                
                layer_results[label]['uas'].append(uas(tree, ud_tree))
                layer_results[label]['uuas'].append(uuas(tree, ud_tree))
            
            
            results['layers'][layer]['uas'] = {
                label: np.mean(layer_results[label]['uas']) for label in ['O', 'I']
            }
            results['layers'][layer]['uuas'] = {
                label: np.mean(layer_results[label]['uuas']) for label in ['O', 'I']
            }
            
            print(f"Layer {layer}")
            print(f"Org UAS: {np.mean(layer_results['O']['uas'])}\tOrg UUAS: {np.mean(layer_results['O']['uuas'])}")
            print(f"Br UAS: {np.mean(layer_results['I']['uas'])}\tBr UUAS: {np.mean(layer_results['I']['uuas'])}\n")
            
        return results
            
    def run(self):
        set_seed()
        print('Extracting matrices...')
        matrix = self.extract_matrices()
        print('Extracting trees...')
        trees = self.extract_trees(matrix)
        return trees
        
        
class AttentionProbe(object):
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        
        if self.args.model in ["facebook/mbart-large-cc25"]:
            self.heads = 16
        else:
            self.heads = 12

        if self.args.cuda:
            self.model.to('cuda')

        self.model = self.model.eval()
    
    def attend(self, sentence, label, root_id, ud_deprels, results):
        
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.encode(sentence)
        tokenized_text = self.tokenizer.convert_ids_to_tokens(indexed_tokens)
        segments_tensor = torch.tensor([0 for _ in tokenized_text])

        # additional check for unknown tokens
        if '[UNK]' in tokenized_text or '<unk>' in tokenized_text:
            return results

        if self.args.cuda:
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            segments_tensor = segments_tensor.to('cuda')
        else:
            tokens_tensor = torch.tensor([indexed_tokens])
            
        # get attention matrices
        with torch.no_grad():
            if self.args.model in ["facebook/mbart-large-cc25"]:
                model_outputs = self.model(tokens_tensor)
                attentions = model_outputs.encoder_attentions
            elif self.args.model in ['bert-base-multilingual-cased']:
                model_outputs = self.model(tokens_tensor, segments_tensor)
                attentions = model_outputs.attentions
            elif self.args.model in ['xlm-roberta-base']:
                model_outputs = self.model(tokens_tensor)
                attentions = model_outputs.attentions

        for k, layer in enumerate(attentions):
            if self.args.cuda:
                attentions_for_layer = layer[0, :, :].cpu().numpy()
            else:
                attentions_for_layer = layer[0, :, :].numpy()
            results[k].append((sentence, label, tokenized_text, root_id, ud_deprels, attentions_for_layer))
                
        return results
    
    def extract_matrices(self):

        out = [[] for _ in range(self.args.layers)]

        for (label, correct_sent, incorrect_sent, cor_root, inc_root, ud_deprels) in tqdm(
                self.data, total=len(self.data)):

            out = self.attend(correct_sent, 'O', cor_root, ud_deprels, out)
            out = self.attend(incorrect_sent, 'I', inc_root, ud_deprels, out)
        
        return out

    def extract_trees(self, results):
        
        layers_trees = {
            layer: {
                head: [] for head in range(self.heads)
            } for layer in range(self.args.layers)
        }

        for layer, state in tqdm(enumerate(results), total=len(results)):

            for s, (sentence, label, tokens, root_id, ud_deprels, matrices) in enumerate(state):
                
                ud_sent = sentence.split()
                
                if self.args.model in ["bert-base-multilingual-cased"]:
                    mapping = map_seq_bert(tokens, ud_sent)
                else:
                    mapping = map_seq_xlmr(tokens, ud_sent)
    
                # put root at position 0 for dependecy extraction
                if self.args.model in ["facebook/mbart-large-cc25"]:
                    perturbed_sent = merge_tokens(tokens, mapping)[:-1]
                elif self.args.model in ["bert-base-multilingual-cased"]:
                    perturbed_sent = merge_tokens(tokens, mapping)[1:-1]
                elif self.args.model in ["xlm-roberta-base"]:
                    perturbed_sent = merge_tokens(tokens, mapping)[1:-1]
            
                perturbed_sent[root_id], perturbed_sent[0] = perturbed_sent[0], perturbed_sent[root_id]

                for head, sent_matrix in enumerate(matrices):
                    
                    # map tokens to words
                    if self.args.model in ["facebook/mbart-large-cc25"]:
                        init_matrix = sent_matrix[:-1, :-1]
                    else:
                        init_matrix = sent_matrix[1:-1, 1:-1]

                    # merge subwords in one row
                    merge_column_matrix = []
                    for i, line in enumerate(init_matrix):
                        new_row = []
                        buf = []
                        for j in range(0, len(line)):
                            buf.append(line[j])
                            if j != len(mapping) - 1:
                                if mapping[j] != mapping[j + 1]:
                                    new_row.append(buf[0])
                            else:
                                new_row.append(buf[0])
                            buf = []
                        merge_column_matrix.append(new_row)

                    # merge subwords in multi rows
                    # transpose the matrix so we can work with row instead of multiple rows
                    merge_column_matrix = np.array(merge_column_matrix).transpose()
                    merge_column_matrix = merge_column_matrix.tolist()
                    final_matrix = []
                    for i, line in enumerate(merge_column_matrix):
                        new_row = []
                        buf = []
                        for j in range(0, len(line)):
                            buf.append(line[j])
                            if j != len(mapping) - 1:
                                if mapping[j] != mapping[j + 1]:
                                    if self.args.subword == 'sum':
                                        new_row.append(sum(buf))
                                    elif self.args.subword == 'avg':
                                        new_row.append((sum(buf) / len(buf)))
                                    elif self.args.subword == 'first':
                                        new_row.append(buf[0])
                            else:
                                new_row.append(buf[0])
                            buf = []
                        final_matrix.append(new_row)

                    # transpose to the original matrix
                    final_matrix = np.array(final_matrix).transpose()

                    # ignore attention to self
                    final_matrix[range(final_matrix.shape[0]), range(final_matrix.shape[0])] = 0

                    # put root at position 0 for tree extraction
                    root = final_matrix[root_id].copy()
                    final_matrix[root_id], final_matrix[0] = final_matrix[0], root

                    final_matrix = np.array(final_matrix, dtype='double')

                    # extract deprels
                    heads, _ = chu_liu_edmonds(final_matrix)
                    deprels = get_deprels(heads, perturbed_sent)
                    
                    assert len(deprels) == len(ud_deprels)
                    
                    layers_trees[layer][head].append((label, deprels, ud_deprels))
       
        return layers_trees
      
    
    def evaluate(self, probe_data):
    
        results = {'layers': {layer: {} for layer in range(self.args.layers)}}
        head_res = {layer: {head: {} for head in range(self.heads)} for layer in range(self.args.layers)}
        
        for layer, heads in tqdm(probe_data.items(), total=self.args.layers, desc='Layer-wise evaluation'):
            
            layer_results = {
                'O': {'uas': [], 'uuas': []}, 
                'I': {'uas': [], 'uuas': []},
            }
        
            for head, trees in heads.items():
                
                head_results = {
                    'O': {'uas': [], 'uuas': []}, 
                    'I': {'uas': [], 'uuas': []},
                }
                
                for label, tree, ud_tree in trees:
                    
                    layer_results[label]['uas'].append(uas(tree, ud_tree))
                    layer_results[label]['uuas'].append(uuas(tree, ud_tree))
                    
                    head_results[label]['uas'].append(uas(tree, ud_tree))
                    head_results[label]['uuas'].append(uuas(tree, ud_tree))
                    
                head_res[layer][head]['uas'] = {
                    label: np.mean(layer_results[label]['uas']) for label in ['O', 'I']
                }
                head_res[layer][head]['uuas'] = {
                    label: np.mean(layer_results[label]['uuas']) for label in ['O', 'I']
                }
                
                print(f"Layer {layer} Head: {head}")
                print(f"Org UAS: {np.mean(head_res[layer][head]['uas']['O'])}\tOrg UUAS: {np.mean(head_res[layer][head]['uuas']['O'])}")
                print(f"Br UAS: {np.mean(head_res[layer][head]['uas']['I'])}\tBr UUAS: {np.mean(head_res[layer][head]['uuas']['I'])}\n")    
            
            results['layers'][layer]['uas'] = {
                label: np.mean(layer_results[label]['uas']) for label in ['O', 'I']
            }
            results['layers'][layer]['uuas'] = {
                label: np.mean(head_results[label]['uuas']) for label in ['O', 'I']
            }
            
            print(f"Layer {layer}")
            print(f"Org UAS: {np.mean(layer_results['O']['uas'])}\tOrg UUAS: {np.mean(layer_results['O']['uuas'])}")
            print(f"Br UAS: {np.mean(layer_results['I']['uas'])}\tBr UUAS: {np.mean(layer_results['I']['uuas'])}\n")
            
        results['heads'] = head_res
        
        return results

    def run(self):
        set_seed()
        print('Extracting matrices...')
        matrix = self.extract_matrices()
        print('Extracting trees...')
        trees = self.extract_trees(matrix)
        return trees
