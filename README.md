# Syntactic Probe
### How Well Do State-of-the-Art Transformer Models Know Syntax?

In this repo you can find the data and the scripts to run evaluation of the structural knowledge of contextualised embeddings.

### Tasks
We present a suit of 3 probing tasks for three languages: English, Swedish and Russian:

1. The (**NShift**) task tests whether contextual representations are sensitive to perturbations in local dependencies.
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

This work deals with the sensitivity of three multilingual transformer-models to syntactic perturbations which can be considered as *structural adversarial probing*. We introduced a set of three tasks, different in the syntactic level of perturbation, for three Indo-European languages: English, Russian and Swedish. We analyse the models' performance on the proposed tasks with the help of perplexity-based methods and probing over attention graphs and intermediate hidden representations produced by the models. 

The results demonstrate, that
1. **The syntactic perturbations are difficult to recover** using only the layer-wise attention weights and intermediate hidden representations produced over sentences with masked positions. While such representations show some ability to capture the syntactic structure, they fail to distinguish between grammatical and perturbed sentences, except for the extreme case of randomly shuffled word order (**RandomShift**). This seems counter-intuitive as the encoders have previously shown to display sufficient word order knowledge. 
2. Despite overall low sensitivity to syntactic perturbations, **the nature of such permutations can, to a degree, influence the models' performance**. All the encoders showed higher sensitivity to more extreme structure perturbations, namely, random shuffling of the words (**RandomShift**) and little to no sensitivity to the inversion within syntactic groups (**Nshift**).
3. **The encoders exhibit different behavior depending on a language**. Contrary to previous works, our probes showed some distinctive features of the encoders depending on the language. Structural information is mostly learnt in a similar manner for Russian and English. **XLM-R** generally captures the properties at the middle-to-higher layers, **M-BART** distributes this information over all the layers and **M-BERT** -- at the higher layers or evenly. For Swedish all the encoders seem to exhibit no special layers that learn sentence structure better. 

All the graphs illustrating the performance of the encoders can be found in the `img` directory.
 
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
