import os, glob, json
import numpy as np

class Saver(object):

    def __init__(self, algorithm):

        directory = os.path.join('./results', algorithm)
        os.makedirs(directory, exist_ok=True)

        exists_results = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
        indices = [int(os.path.basename(tmp).split("_")[1]) for tmp in exists_results]
        
        run_id = str(0) if len(indices)==0 else np.max(indices) + 1
        self.experiment_dir = os.path.join(directory, f'experiment_{run_id}')
        os.makedirs(self.experiment_dir, exist_ok=True)

    def save_experiment_config(self, args):
        self.save_args = args.__dict__.copy()
        if 'cuda' in self.save_args:
            del(self.save_args['cuda'])

        with open(os.path.join(self.experiment_dir, 'arg_parser.txt'), 'w') as f:
            json.dump(self.save_args, f, indent=2)
        
        f.close()
        