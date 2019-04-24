"""
Run the experiments for a Michelle Obama
and Ellen datasets, to verify the model
parameters selected for Trump data.

@author Victoria Proetsch
"""
from train import load_and_shape_data, run_experiment

# --------------------------------
# Load data for word models
# --------------------------------

train, val, test, full, vocab_size, index_word = load_and_shape_data('data/Ellen_tweetdata.txt', char_model=False, seq_len=8)

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='validation-ellen',
              epochs= 20,
              char_model=False, 
              seq_len=8,
              depth=1, 
              dropout=0,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

# --------------------------------
# Load data for word models
# --------------------------------

train, val, test, full, vocab_size, index_word = load_and_shape_data('data/Michelleobama_tweetdata.txt', char_model=False, seq_len=8)

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='validation-michelle',
              epochs= 20,
              char_model=False, 
              seq_len=8,
              depth=1, 
              dropout=0,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)