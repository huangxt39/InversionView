# A Brief Description of Code

## Folders

- `./counting` 
  1. `train.py` train the character counting model, as well as make up the data
  2. `intervention.py` activation patching experiments
  3. `visualize.ipynb` results visualization code
- `./ioi` 
  1. `make_decoder_train_data.py` contains code to make up the data
  2. `DLA.py` implements the DLA experiments
- `./addition` 
  1. `train.py` train the 3 digit addition model, as well as make up the data
  2. `intervention.py` main activation patching experiments
  3. `interventionPlus.py` activation patching experiments for the "+" sign
  4. `visualize.ipynb` results visualization code
- `./factual` 
  1. `find_heads_attribution.py` find 25 most important heads in upper layers
  2. `make_data_part1.py` select text from COUNTERFACT and BEAR that would "activate" each head (do not attend to BOS too much) at the END position 
  3. `make_data_part2.py` select text from miniPile that would "activate" each head
  4. `cal_freq.py` calculate token frequency over miniPile
- `./decoder`
  1. `model.py` defines the model architecture
  2. `train.py` train the decoder
  3. `cache_generation.py` generate samples using decoder, but not in a visualized form, need to be transferred to streamlit app
  4. `run.sh` commands to train decoder and generate samples
  5. `utils.py` `generate.py` functions used by other files
  6. `cache_attention.py` used to save attention patterns
  7. `scatter_completeness.py` `scatter_completeness_plot.py` draw scatter plots to verify the completeness
- `./training_outputs`
  contains the model checkpoint of the probed model for counting and addition task, so the results are reproduceable
- `./LLM` contains prompts and code used to automatically generate interpretation with LLMs
- `./webAPP` contains source code for our web application


## Steps to run

1. Go to `./ioi` and run `make_decoder_train_data.py` to generate data for ioi task. You don't need to do this for counting and addition task. To run factual recall experiment, first download COUNTERFACT and BEAR data (the links are in `./factual/make_data_part1.py`), then go to `./factual` and run `make_data_part1.py` and `make_data_part2.py` sequentially.
2. Go to `./decoder` and check `run.sh` pick a task you are interested and train the decoder. For example
   `python train.py --probed_task counting --rebalance 6.0 --save_dir $dir_name --batch_size 256 --num_epoch 100 --data_per_epoch 1000000 --num_test_rollout 200 > ./data_and_model/counting.txt`
3. In `run.sh` it also contains command for generating preimage samples using decoder. For example, 
   `python cache_generation.py --probed_task counting`
   and the generation will appear in `./training_outputs`
   The best way to check the generated samples is to go into `./webAPP` folder and do `streamlit run InversionView.py`
