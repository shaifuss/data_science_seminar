import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import re

from spellchecker import SpellChecker

def regex_filter(sentence):
    # fix missing delimiter - i.e deepDishPizza
    sentence = re.sub(r'([a-z])([A-Z])', r'\1\. \2', sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'&gt|&lt', ' ', sentence)
    # fix letter repetition (if more than 2)
    sentence = re.sub(r'([a-z])\1{2,}', r'\1', sentence)
    # fix non-word repetition (if more than 1)
    sentence = re.sub(r'([\W+])\1{1,}', r'\1', sentence)
    # string * as delimiter
    sentence = re.sub(r'\*|\W\*|\*\W', '. ', sentence)
    # xxx[?!]. -- > xxx.
    sentence = re.sub(r'\W+?\.', '.', sentence)
    # [.?!] --> [.?!] xxx
    sentence = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', sentence)
    # fix phrase repetition
    sentence = re.sub(r'(.{2,}?)\1{1,}', r'\1', sentence)

    return sentence.strip()
    
# remove numbers and punctuation marks
def filter_punctuation(word_list):
    return [word for word in word_list if word.isalpha()]

# remove unimportant connective words such as "and", "the", etc
def filter_stopwords(word_list):
  return [word for word in word_list if word not in stopwords.words('english')]

# keep only nouns
def retain_nouns(word_list):
    return [word for (word, pos) in nltk.pos_tag(word_list) if pos[:2] in ['NN']]

# normlize for part of speech
def stem_words(word_list):
  ps = PorterStemmer()
  return [ps.stem(word) for word in word_list]

def fix_spelling(word_list):
  spell = SpellChecker()
  return [spell.correction(word) for word in word_list]
  
  
def preprocess_words(text):
  word_list = word_tokenize(text)
  word_list = filter_punctuation(word_list)
  word_list = fix_spelling(word_list) 
  word_list = filter_stopwords(word_list)
  word_list = retain_nouns(word_list)
  return stem_words(word_list)
  
def preprocess(reviews, samp_size=None):
  if not samp_size:
        samp_size = 1000

  start = time.time()
  print('Stage 1: Preprocess raw review texts')
  texts = []  
  token_lists = []  
  idx_in = []
  indicies = np.random.choice(len(reviews), samp_size)
  for i in indicies:
      text = regex_filter(reviews[i])
      token_list = preprocess_words(text)
      if token_list:
        idx_in.append(i)
        texts.append(text)
        token_lists.append(token_list)
         
  end = time.time()
  print("Preprocessing {} reviews took {} minutes".format(len(indicies), str((end - start)/60)))
  return texts, token_lists, idx_in

if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2])
