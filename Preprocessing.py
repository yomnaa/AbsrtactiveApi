import subprocess
import os
import glob
import re
from nltk.tokenize.stanford import StanfordTokenizer
class Preprocessor:
  def __init__(self):
    self.dm_single_close_quote = u'\u2019'  # unicode
    self.dm_double_close_quote = u'\u201d'
    self.END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', self.dm_single_close_quote, self.dm_double_close_quote,
                  ")"]  # acceptable ways to end a sentence

    # We use these to separate the summary sentences in the .bin datafiles
    self.SENTENCE_START = '<s>'
    self.SENTENCE_END = '</s>'
    self.tokenizer=StanfordTokenizer('stanford-postagger.jar', options={"tokenizeNLs": True})
  def tokenize(self,article):
    return self.tokenizer.tokenize(article)


  def fix_missing_period(self,line):
    """Adds a period to a line that is missing a period"""


    if line == "": return line
    if line[-1] in self.END_TOKENS: return line
    # print line[-1]
    return line + " ."


  def adjust_article(self,article):
    #takes the article t

    # Lowercase everything
    lines = [line.lower() for line in article]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [self.fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []

    for idx,line in enumerate(lines):
      if line == "":
        continue # empty line
      else:
        article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    # abstract = ' '.join(["%s %s %s" % (self.SENTENCE_START, sent, self.SENTENCE_END) for sent in highlights])

    return article
# preprocessor=Preprocessor()
# directory='/media/yomna/Life/Graduationprojectdetermination/codedandtutorials/cnndailymaster/cnn-dailymail-master/'
# preprocessor.tokenize('/media/yomna/Life/Graduationprojectdetermination/codedandtutorials/cnndailymaster/cnn-dailymail-master/','article.txt')
# tokenized=open(directory+'tokenized.txt','r').readlines()
# preprocessor.adjust_article(tokenized)


