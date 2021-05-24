# Syntactic Probe
### How Well Do State-of-the-Art Transformer Models Know Syntax?

In this repo you can find the data and the scripts to run evaluation of the structural knowledge of contextualised embeddings.

### Tasks
We present a suit of 3 probing tasks for three languages: English, Swedish and Russian:

1. The (**NgramShift**) task tests whether contextual representations are sensitive to perturbations in local dependencies.
2. The (**ClauseShift**) task probes  the encoder sensitivity to long-range dependencies by perturbing a sentence at the clause-level.
3. The (**RandomShift**) task test the ability of the encoder to restore the tree of a sentence corrupted by random shuffling of the words.

### Models
The code supports three models, available via the HuggingFace library:

1. **M-BERT** [(Devlin et al. 2019)](https://arxiv.org/abs/1810.04805), a transformer mo)el of the encoder architecture, trained on multilingual Wikipedia data using the Masked LM (MLM) and Next Sentence Prediction pre-training objectives.
2. **XLM-RoBERTa** [(Conneau et al. 2020)](https://arxiv.org/abs/1911.02116), a large language model trained on monolingual CommonCrawl with the multilingual MLM objective.
3. **M-BART** [(Liu et al. 2020)](https://arxiv.org/abs/2001.08210), a sequence-to-sequence denoising auto-encoder. For our experiment we use only the encoder, pretrained on the monolingual corpora with the BART objective.

### Experiments

1. **Attention Probe** [(Htut et al., 2019)](): extract and evaluate syntax trees from attention matrices of the encoder.
2. **Perturbed Probe** [(Wu et al., 2020)](https://arxiv.org/pdf/2004.14786https://arxiv.org/abs/1911.12246?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529.pdf): extract and evaluate syntax trees obtained from the token impact matrices from the hidden states of the encoder.
3. **Acceptability Measures**: perplexity-based method to evaluate encoders awareness of sentence grammaticality. We use log probability (LP)
 and its two variants, Mean LP and Pen LP, normalised by length.
 
 ### Results
 
 TBD
 
## Setup and Usage
Clone this repo and install all the dependencies:
```
git clone https://github.com/evtaktasheva/syntactic-probe
cd syntactic-probe
sh install_tools.sh 
```
1. Example run of the probe:
```
from probing.args import Args
from probing.experiment import Experiment

args = Args()
args.model = 'bert-base-multilingual-cased' #xlm-roberta-base #facebook/mbart-large-cc25
args.probe_tasks = ['en_ngram_shift', 'ru_ngram_shift', 'sv_ngram_shift']
args.prober = 'perturbed'  #'attention'

experiment = Experiment(args=args)
experiment.run()
```
2. To calculate acceptability measures:
```
from probing.args import Args
from probing.experiment import LogProb

args = Args()
args.model = 'bert-base-multilingual-cased' #xlm-roberta-base #facebook/mbart-large-cc25
args.probe_tasks = ['en_ngram_shift', 'ru_ngram_shift', 'sv_ngram_shift']
args.prober = 'logprob'

experiment = LogProb(args=args)
experiment.run()
```
