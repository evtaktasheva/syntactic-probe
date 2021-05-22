from probing.extractor import PerturbedProbe, AttentionProbe
from probing.models_utils import LoadModels
from probing.utilities import DataLoader, save_results


class Experiment:
    def __init__(self, args, max_len=-1):
        self.args = args
        self.max_len = max_len

    def run(self):
        print('Loading models...')
        model_loader = LoadModels(self.args.model, self.args.prober)
        model = model_loader.load_model()
        tokenizer = model_loader.load_tokenizer()

        for probe_task in self.args.probe_tasks:
            
            print('* * ' * 30)
            print(f'Running {self.args.prober} on {probe_task}...')
            
            print('Loading data...')
            data = DataLoader(
                probe_task,
                max_len=self.max_len
            ).load_data()

            results = {
                'prober': self.args.prober,
                'model': self.args.model,
                'task': probe_task,
                'subword': self.args.subword,
                'tree_metric': self.args.metric
            }
            
            if self.args.prober == 'attention':
                prober = AttentionProbe(data=data, model=model, tokenizer=tokenizer, args=self.args)
                probe_results = prober.run()
                
            elif self.args.prober == 'perturbed':
                prober = PerturbedProbe(data=data, model=model, tokenizer=tokenizer, args=self.args)
                probe_results = prober.run()
    
            probe_results = prober.evaluate(probe_results)   
            results.update(probe_results)
            
            save_results(prober=self.args.prober,
                         model=self.args.model,
                         task=probe_task,
                         data=results)