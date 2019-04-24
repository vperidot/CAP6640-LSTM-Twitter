"""
Functions to define, train, and evaluate the model

@author Victoria Proetsch
"""
import time
import os

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from nltk.translate.bleu_score import sentence_bleu

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

from model import make_model
from nlp_preprocessing import load_data_indexes, load_data_chars, vec_to_text, create_sequence_next_pairs, train_val_test_split

def do_inference(seed, max_steps, model, seq_len=8):
  """
  Given a seed text, predict the remaining tweet.
  Will generate until either END token or max_steps.
  """
  
  prediction = []
  p = -1
  while (max_steps > 0) and (p != 2):
    output = model.predict(seed)
    p = np.argmax(output)
    new_seed = np.append(seed[0, 1:], p).reshape(1,seq_len)
    #print('new_seed: {}'.format(new_seed))
    seed = new_seed
    prediction.append(p)
    max_steps -= 1
    
  return prediction

def load_and_shape_data(filename, char_model=False, seq_len=8):
  """
  Load Data
  Chop it up into chunks of seq_len
  """

  # Load it
  if char_model:
    data_loader = load_data_chars
  else:
    data_loader = load_data_indexes

  train, word_index = data_loader(filename)
  index_word = {i: w for w, i in word_index.items()}
  vocab_size = len(word_index)

  # First let's look at some training examples.
  # print(train[0])
  # print(vec_to_text(train[0], index_word))
  # print(vec_to_text(train[1], index_word))
  # print(vec_to_text(train[2], index_word))

  # Turn it into sequence/next pairs
  X, y = create_sequence_next_pairs(train, seq_len=seq_len)

  # Split it into train/val/test
  (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y, train_pct=0.7, val_pct=0.15)

  #print(X_train.shape, y_train.shape)

  # response variable should be one-hot
  y_train = to_categorical(y_train, num_classes=vocab_size)
  y_val = to_categorical(y_val, num_classes=vocab_size)
  y_test = to_categorical(y_test, num_classes=vocab_size)

  return (X_train, y_train), (X_val, y_val), (X_test, y_test), train, vocab_size, index_word

def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, full, seq_len=8, epochs=20,
                  depth=3, dropout=0.25, batch_norm=False, char_model=False, vocab_size=2500, 
                  vocab_dict=None, exp_name=''):

  """
  Run an experiment given a set of parameters.
  Creates, trains, and evaluates a model.
  Results will be saved in a directory named models/{exp_name}/
  """
  #---------------------
  # Model Training
  #---------------------

  # create the model from experiment parameters
  model = make_model(sequences=False, layers=depth, dropout=dropout, vocab_size=vocab_size, seq_len=seq_len)
  model.summary()

  # make directories to save training logs and model results
  os.makedirs('training_logs/'+exp_name+'/', exist_ok=True)
  os.makedirs('models/'+exp_name+'/', exist_ok=True)

  # add tensorboard callback so we can watch the training process
  tensorboard = TensorBoard(log_dir="training_logs/{}".format(exp_name),
                            update_freq='batch')

  # train the model
  history = model.fit(X_train, y_train,
            batch_size=32, epochs=epochs, shuffle=False,
            validation_data=(X_val, y_val),
            callbacks=[tensorboard])

  # save model
  model.save_weights('models/'+exp_name+'/'+exp_name+'_weights.h5')

  #--------------------------
  # Quantitative Results
  #--------------------------

  plt.style.use('seaborn')

  # training & validation accuracy
  fig, ax = plt.subplots(figsize=(15, 5))
  ax.plot(history.history['acc'], label='train accuracy')
  ax.plot(history.history['val_acc'], label='valid. accuracy')
  plt.title('{} : Accuracy'.format(exp_name), fontsize=18)
  plt.legend(fontsize=16)
  plt.savefig('models/'+exp_name+'/accuracy_curve.png')

  # training & validation loss
  fig, ax = plt.subplots(figsize=(15, 5))
  ax.plot(history.history['loss'], label='train loss')
  ax.plot(history.history['val_loss'], label='valid. loss')
  plt.title('{} : Loss'.format(exp_name), fontsize=18)
  plt.legend(fontsize=16)
  plt.savefig('models/'+exp_name+'/loss_curve.png')

  # test loss/accuracy
  print('evaluating model...')
  results = model.evaluate(X_test, y_test, batch_size=32)
  test_loss, test_accuracy = results[0], results[1]
  print(results)

  #--------------------------
  # Qualitative Results
  #--------------------------

  # pull out all sentence starters from test data
  #seeds = [seed for seed in X_test if seed[0]==1 ]
  references = [f for f in full if len(f) >= seq_len]
  references = references[:200]
  print(references[:3])
  seeds = [r[:seq_len] for r in references]
  print(seeds[:3])
  # grab the first 200 only (makes calculating the score much faster)
  # seeds = seeds[:200]
  # print a few as a sanity check
  # print('examples from {} seeds:'.format(len(seeds)))
  # print(vec_to_text(seeds[0], index_word))
  # print(vec_to_text(seeds[1], index_word))
  # print(vec_to_text(seeds[2], index_word))

  # setup for predictions
  predicted_tweets = []
  bleu_1, bleu_2, bleu_3, bleu_4 = 0., 0., 0., 0.
  print('calculating bleu score...')

  # calucate a bleu score for each prediction
  # while collecting predicted tweets
  for s in range(len(seeds)):

    seed = np.array(seeds[s]).reshape((1, seq_len))
    max_steps = 132 if char_model else 32 
    vec_pred = do_inference(seed, max_steps, model, seq_len=seq_len)
    predicted_tweet = list(seed[0]) + list(vec_pred)
    predicted_tweets.append(predicted_tweet)
    bleu_1 += sentence_bleu([references[s]], predicted_tweet, weights=(1,0,0,0))
    bleu_2 += sentence_bleu([references[s]], predicted_tweet, weights=(0,1,0,0))
    bleu_3 += sentence_bleu([references[s]], predicted_tweet, weights=(0,0,1,0))
    bleu_4 += sentence_bleu([references[s]], predicted_tweet, weights=(0,0,0,1))

  # average the bleu score
  bleu_1_avg = bleu_1/len(seeds)
  bleu_2_avg = bleu_2/len(seeds)
  bleu_3_avg = bleu_3/len(seeds)
  bleu_4_avg = bleu_4/len(seeds)
  print('BLEU-1 SCORE: {}'.format(bleu_1_avg))
  print('BLEU-2 SCORE: {}'.format(bleu_2_avg))
  print('BLEU-3 SCORE: {}'.format(bleu_3_avg))
  print('BLEU-4 SCORE: {}'.format(bleu_4_avg))

  # save test results
  output  = 'BLEU-1 SCORE : {}\n'.format(bleu_1_avg)
  output += 'BLEU-2 SCORE : {}\n'.format(bleu_2_avg)
  output += 'BLEU-3 SCORE : {}\n'.format(bleu_3_avg)
  output += 'BLEU-4 SCORE : {}\n'.format(bleu_4_avg)
  output += 'TEST LOSS    : {}\n'.format(test_loss)
  output += 'TEST ACC.    : {}'.format(test_accuracy)

  with open('models/'+exp_name+'/predicted_tweets.txt', 'w') as f:
    if char_model:
      f.write( ''.join([vec_to_text(tweet, vocab_dict) for tweet in predicted_tweets]) )
    else:
      f.write( '\n'.join([vec_to_text(tweet, vocab_dict) for tweet in predicted_tweets]) )

  with open('models/'+exp_name+'/score.txt', 'w') as f:
    f.write( output )
