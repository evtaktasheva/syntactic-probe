from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig
from transformers import BertTokenizer, MBartTokenizer, XLMRobertaTokenizer


class LoadModels(object):
    def __init__(self, model, prober):
        self.model = model
        self.prober = prober

    def load_model(self):
        config = AutoConfig.from_pretrained(
            self.model,
            output_attentions=(True if self.prober == 'attention' else False),
            output_hidden_states=(True if self.prober == 'perturbed' else False)
        )
        if self.prober == 'logprob':
            model = AutoModelForMaskedLM.from_pretrained(self.model)
        else:
            model = AutoModel.from_config(config)
        return model

    def load_tokenizer(self):
        if self.model in ["xlm-roberta-base"]:
            return XLMRobertaTokenizer.from_pretrained(self.model, strip_accents=False)
        elif self.model in ['bert-base-multilingual-cased']:
            return BertTokenizer.from_pretrained(self.model, strip_accents=False)
        elif self.model in ['facebook/mbart-large-cc25']:
            return MBartTokenizer.from_pretrained('facebook/mbart-large-cc25', strip_accents=False)
