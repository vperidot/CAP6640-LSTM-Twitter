"""
Run the experiments for a word-based model

@author Victoria Proetsch
"""
from train import load_and_shape_data, run_experiment

# --------------------------------
# Load data for word models
# --------------------------------

train, val, test, full, vocab_size, index_word = load_and_shape_data('data/Trump_tweetdata.txt', char_model=False, seq_len=8)

# --------------------------------
# Depth Experiments
# --------------------------------

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='words-depth-1',
              epochs= 20,
              char_model=False, 
              seq_len=8,
              depth=1, 
              dropout=0,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='words-depth-2',
              epochs= 20,
              char_model=False, 
              seq_len=8,
              depth=2, 
              dropout=0,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='words-depth-3',
              epochs= 20,
              char_model=False, 
              seq_len=8,
              depth=3, 
              dropout=0,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

# --------------------------------
# Dropout Experiments
# Run on a depth of 1, determined 
# by above experiments
# --------------------------------

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='words-dropout-25',
              epochs= 20,
              char_model=False, 
              seq_len=8,
              depth=1, 
              dropout=0.25,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='words-dropout-50',
              epochs= 20,
              char_model=False, 
              seq_len=8,
              depth=1, 
              dropout=0.5,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)

run_experiment(train[0], train[1],
              val[0], val[1], 
              test[0], test[1], 
              full,
              exp_name='words-dropout-65',
              epochs= 20,
              char_model=False, 
              seq_len=8,
              depth=1, 
              dropout=0.65,
              batch_norm=False, 
              vocab_size=vocab_size,
              vocab_dict=index_word)