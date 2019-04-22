"""
Run the experiments for a character-based model

@author Victoria Proetsch
"""
from train import load_and_shape_data, run_experiment

# --------------------------------
# Load data for character models
# --------------------------------

train, val, test, full, vocab_size, index_word = load_and_shape_data('data/Trump_tweetdata.txt', char_model=True, seq_len=12)

# --------------------------------
# Depth Experiments
# --------------------------------

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='char-depth-1',
              epochs= 20,
              char_model=True, 
              seq_len=12,
              depth=1, 
              dropout=0,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='char-depth-2',
              epochs= 20,
              char_model=True, 
              seq_len=12,
              depth=2, 
              dropout=0,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='char-depth-3',
              epochs= 20,
              char_model=True, 
              seq_len=12,
              depth=3, 
              dropout=0,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

