
def create_text_file_with_certain_text(text, file_name):
    with open(file_name, 'w') as f:
        f.write(text)
        
        
def create_text(start):
    #random number generator from 8080 to 8099

    text = """#!/bin/bash

#SBATCH --mail-type=NONE
#SBATCH --output=./sbatch_logs/{}%j.log
#SBATCH --nodes=1
#SBATCH --partition=cpuq
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --job-name={}_Allen
#SBATCH --time=96:00:00

source $HOME/.bashrc
conda activate microsim_pytorch
python sim_vera_allen.py --start {}
""".format(start, start, start)

    return text



for i in range(0000, 6000, 100):
    i_ = str(i)
    text = create_text(i)
    create_text_file_with_certain_text(text, 'sbatches/{}.sbatch'.format(i_))

    
