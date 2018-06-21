# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""

import os
import time
import tensorflow as tf
import beam_search
import data
import json
import util
import logging
import numpy as np

# FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, vocab,single_pass,hps,pointer_gen,log_root):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    # self._batcher = batcher
    self._vocab = vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())
    self.single_pass=single_pass
    self. max_dec_steps=hps.max_dec_steps
    self.min_dec_steps=hps.min_dec_steps
    self.max_dec_steps=hps.max_dec_steps
    self.beam_size=hps.beam_size
    self.pointer_gen=pointer_gen

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess,log_root)
    print ckpt_path
    #
    # if FLAGS.single_pass:
    #   # Make a descriptive decode directory name
    #   ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
    #   self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
    #   if os.path.exists(self._decode_dir):
    #     raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)
    #
    # else: # Generic decode dir name
    #   self._decode_dir = os.path.join(FLAGS.log_root, "decode")
    #
    # # Make the decode dir if necessary
    # if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

    #This code has been commented by me

    # if FLAGS.single_pass:
    #   # Make the dirs to contain output written in the correct format for pyrouge
    #   self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
    #   if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
    #   self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
    #   if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
  def setBatcher(self,batcher):
    self._batcher = batcher

  def decode(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    out_num=0
    summaries=[]
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert self.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")

      ##I commented those lines
        # tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        # results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        # rouge_log(results_dict, self._decode_dir)
        print out_num
        return summaries

      original_article = batch.original_articles[0]  # string
      # I commented those lines
      # original_abstract = batch.original_abstracts[0]  # string
      # original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

      article_withunks = data.show_art_oovs(original_article, self._vocab) # string
      # I commented this line
      # abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

      # Run beam search to get best Hypothesis
      best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch,self.beam_size,self.max_dec_steps,self.min_dec_steps)

      # Extract the output ids from the hypothesis and convert back to words
      output_ids = [int(t) for t in best_hyp.tokens[1:]]
      decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if self.pointer_gen else None))

      # Remove the [STOP] token from decoded_words, if necessary
      try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
      except ValueError:
        decoded_words = decoded_words
      decoded_output = ' '.join(decoded_words) # single string

      if self.single_pass:
        summaries.append(decoded_output)
        # open('s'+str(out_num)+'.txt','w').write(decoded_output)
        # with open('output'+str(out_num)+'.txt','w') as output:
          # output.write(original_article+'\n*******************************************\n\n'+decoded_output)
        out_num+=1
        print out_num
        #this line is commented by me
        # self.write_for_rouge(original_abstract_sents, decoded_words, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
        counter += 1 # this is how many examples we've decoded


def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s



