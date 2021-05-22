from probing.models_utils import LoadModels
from probing.utilities import *
from scipy.special import softmax
from scipy.stats.mstats import pearsonr as corr
import torch
import math


class LogProb(object):
    
    def __init__(self, args, max_len=-1):
        self.args = args
        self.max_len = max_len

    def model_score(self, tokenize_input, model, tokenizer):
        
        indexed_tokens = tokenizer.encode(tokenize_input)
        tokenize_combined = tokenizer.convert_ids_to_tokens(indexed_tokens)
        
        batched_indexed_tokens = []
        batched_segment_ids = []

        for i in range(len(indexed_tokens)-2):
    
            # Mask a token that we will try to predict back
            masked_index = i + 1
            indexed_masked = indexed_tokens.copy()
            indexed_masked[masked_index] = tokenizer.mask_token_id

            # Define sentence A and B indices associated to 1st and 2nd sentences
            segment_ids = [0]*len(indexed_masked)
    
            batched_indexed_tokens.append(indexed_tokens)
            batched_segment_ids.append(segment_ids)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(batched_indexed_tokens)
        segment_tensor = torch.tensor(batched_segment_ids)
        
        if self.args.cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            segment_tensor = segment_tensor.to('cuda')
        
        # Predict all tokens
        if self.args.model in ["facebook/mbart-large-cc25"]:
            with torch.no_grad():
                outputs = model(tokens_tensor)
                predictions = outputs.encoder_last_hidden_state
        else:
            with torch.no_grad():
                outputs = model(tokens_tensor, token_type_ids=segment_tensor)
                predictions = outputs[0]
        
        # go through each word and sum their logprobs
        lp = 0.0
        for i in range(len(tokenize_input)):
            masked_index = i + 1
            predicted_score = predictions[i, masked_index]
            predicted_prob = softmax(predicted_score.cpu().numpy())
            lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]])
            
        return lp
        
    def run(self):
        
        print('Loading models...')
        model_loader = LoadModels(self.args.model, self.args.prober)
        model = model_loader.load_model()
        tokenizer = model_loader.load_tokenizer()
        
        if self.args.cuda:
            model = model.to('cuda')
        model = model.eval()
    
        for probe_task in self.args.probe_tasks:
            
            print('* * ' * 30)
            print(f'Calculating logprob on {probe_task}...')
            
            print('Loading data...')
            data = DataLoader(
                probe_task,
                max_len=self.max_len
            ).load_data()
            
            results = {
                label: {
                    metric: [] 
                    for metric in ['lp', 'mean_lp', 'pen_lp']
                } for label in ['O', 'I']
            }
            
            lps = []
            mean_lps = []
            pen_lps = []
            
            # loop through each sentence and compute system scores
            for sent_id, (text, label, root_id, ud_deprels) in tqdm(enumerate(data), total=len(data)):
                
                if label == 'O':
                    continue
                
                tokenize_input = tokenizer.tokenize(text)
                text_len = len(tokenize_input)
        
                # compute sentence logprob
                lp = self.model_score(tokenize_input, model=model, tokenizer=tokenizer)
        
                # acceptability measures
                penalty = ((5+text_len)**0.8 / (5+1)**0.8)
                results[label]['lp'].append(lp)
                results[label]['mean_lp'].append(lp / text_len)
                results[label]['pen_lp'].append( lp / penalty )
                
                
                correct_sent = ' '.join([dep[0] for dep in ud_deprels])
                tokenize_input = tokenizer.tokenize(correct_sent)
                text_len = len(tokenize_input)
                
                # compute sentence logprob
                lp = self.model_score(tokenize_input, model=model, tokenizer=tokenizer)
        
                # acceptability measures
                penalty = ((5+text_len)**0.8 / (5+1)**0.8)
                results['O']['lp'].append(lp)
                results['O']['mean_lp'].append(lp / text_len)
                results['O']['pen_lp'].append( lp / penalty )

            results['model'] = self.args.model
            
            save_results(prober='logprob',
                         model=self.args.model,
                         task=probe_task,
                         data=results)
