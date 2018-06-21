import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
# import BeautifulSoup
from bs4 import BeautifulSoup
from urllib2 import urlopen
import Preprocessing
import glob
import util
from tensorflow.python import debug as tf_debug
# FLAGS = tf.app.flags.FLAGS
#
# # Where to find data
# # tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
# tf.app.flags.DEFINE_string('vocab_path','/media/yomna/Life/Graduationprojectdetermination/codedandtutorials/cnn-dailymail-master/finished_files/vocab', 'Path expression to text vocabulary file.')
#
# # Important settings
# tf.app.flags.DEFINE_string('mode', 'decode', 'must be one of train/eval/decode')
# tf.app.flags.DEFINE_boolean('single_pass', True, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
#
# # Where to save output
# tf.app.flags.DEFINE_string('log_root', '/media/yomna/Life/Graduationprojectdetermination/codedandtutorials/pretrained get to the point code/trained_model/model', 'Root directory for all logging.')
# tf.app.flags.DEFINE_string('exp_name', 'model', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
#
# # Hyperparameters
# tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
# tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
# tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
# tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
# tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
# tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
# tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
# tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
# tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
# tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
# tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
# tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
# tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
#
# # Pointer-generator or baseline model
# tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
#
# # Coverage hyperparameters
# tf.app.flags.DEFINE_boolean('coverage', True, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
# tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
#
# # Utility flags, for restoring and changing checkpoints
# tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
# tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')
#
# # Debugging. See https://www.tensorflow.org/programmers_guide/debugger
# tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

class Summarizer:
    def __init__(self,vocab_path,log_root):
        self.pointer_gen=True
        self.single_pass=True
        self.batch_size = self.beam_size=4
        self.vocab_size=50000
        self.vocab_path=vocab_path
        self.log_root=log_root
        # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
        hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                       'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage',
                       'cov_loss_wt',
                       'pointer_gen']
        hps_dict = {
            'mode': 'decode',
            'lr': 0.15,
            'adagrad_init_acc': 0.1,
            'rand_unif_init_mag': 0.02,
            'trunc_norm_init_std': 1e-4,
            'max_grad_norm': 2.0,
            'hidden_dim': 256,
            'emb_dim': 128,
            'batch_size': self.batch_size,
            'max_dec_steps': 100,
            'max_enc_steps': 400,
            'coverage': 1,
            'cov_loss_wt': 1.0,
            'pointer_gen': True,
            'min_dec_steps': 35,
            'beam_size': self.beam_size
        }



        self.hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
        self.vocab = Vocab(self.vocab_path, self.vocab_size)
        tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
        # If in decode mode, set batch_size = beam_size
        # Reason: in decode mode, we decode one example at a time.
        # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.

        #
        decode_model_hps = self.hps  # This will be the hyperparameters for the decoder model
        decode_model_hps = self.hps._replace(
            max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries


        tf.set_random_seed(111)  # a seed value for randomness
        self.model = SummarizationModel(decode_model_hps, self.vocab,self.log_root)
        self.decoder = BeamSearchDecoder(self.model, self.vocab,True,self.hps,self.pointer_gen,self.log_root)


    def summarize(self,articles):
        self.batcher = Batcher(articles, self.vocab, self.hps, single_pass=self.single_pass)
        self.decoder.setBatcher(self.batcher)
        return self.decoder.decode()  # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)






