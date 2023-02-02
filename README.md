# Meta-Learning Framework for Schema Matching
_Elizabeth Witten, Fall 2022_

Code for CS7290 final project, "Meta-Learning Framework for Schema Matching"

## Abstract
Given the large and hetergeneous nature of data lakes, dataset discovery is a difficult task on its own. Schema matching---the task of finding matching pairs of columns between a source and target schema---is an important component to finding relationships between two tables. In this work, a novel meta-learned schema matching technique is introduced and evaluated against other instance-based schema matching techniques. A detailed look at the results offers insight into where meta-learning and data augmentation can improve matching effectiveness.

## Repository Organization
- **docs** folder contains additional written material, including the project report and presentation slides.
- **notebooks** folder contains python notebooks used for data generation and Valentine testing and the output folders with the data used in the matching experiments.
- **results** folder contains some graphs of the matching experiments results.
- **rotom** folder contains the Rotom source code, pulled from [here](https://github.com/megagonlabs/rotom) and adapted for schema matching.

## Setup
1. Clone this repository
2. In the repository directory, create and activate a virtual environment for this project
3. Follow the setup directions in `rotom/README.md` to install the requirements for Rotom

## Reproducibility

### 1. All train, validation, and test data have been provided in `rotom/data/em/SANTOS-XS_notest`.

  > Note: do not rename these directories or files, or the Rotom script will not be able to find them.
  
  - `train.txt` contains the serialized training data
  - `train.txt.augment.jsonl` contains the augmented training data produced by the InvDA seq2seq model
  - `valid.txt` contains the serialized validation data
  - `test.txt` contains dummy test data for running the training script
  - `all_test.txt` contains all of the serialized test data
  - `individual_tests/*.txt` contains the serialized test data, split by table set
  - `individual_tests/out/*.csv` contains the test results, split by schema matching nodel
  
### 2. In the `rotom/` directory, train the four Rotom-based models with the following commands:

|     Model                  |     Run                                                                                                                                                                                                                                                     |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     Rotom w/ SSL           | CUDA_VISIBLE_DEVICES=0 python train_any.py --task em_SANTOS-XS+OpenData-Verbatim --size 300 --logdir results_em/ --finetuning --batch_size 32 --lr 3e-5 --n_epochs 20 --max_len 128 --fp16 --lm roberta --da auto_filter_weight --balance --run_id 0 --save_model        |
|     Rotom w/o SSL          | CUDA_VISIBLE_DEVICES=0 python train_any.py --task em_SANTOS-XS+OpenData-Verbatim --size 300 --logdir results_em/ --finetuning --batch_size 32 --lr 3e-5 --n_epochs 20 --max_len 128 --fp16 --lm roberta --da auto_filter_weight_no_ssl --balance --run_id 0 --save_model |
|     InvDA + LM Fine-Tuning | CUDA_VISIBLE_DEVICES=0 python train_any.py --task em_SANTOS-XS+OpenData-Verbatim --size 300 --logdir results_em/ --finetuning --batch_size 32 --lr 3e-5 --n_epochs 20 --max_len 128 --fp16 --lm roberta --da invda --balance --run_id 0 --save_model                     |
|     LM Fine-Tuning         | CUDA_VISIBLE_DEVICES=0 python train_any.py --task em_SANTOS-XS+OpenData-Verbatim --size 300 --logdir results_em/ --finetuning --batch_size 32 --lr 3e-5 --n_epochs 20 --max_len 128 --fp16 --lm roberta --da None --balance --run_id 0 --save_model                      |

- The trained models will be saved to the `rotom/` directory.

### 3. Open the `rotom/load_and_test.ipynb` notebook to run tests.
  - Under the **Set Parameters Here** section:
    - Set the parameters used to train the model to test
    - Set the path to the directory containing the test files (`data/em/SANTOS-XS/individual tests/`)
  - Run the notebook, which loads the model and records the results for each test file
