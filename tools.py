import os
import nltk
import re
from nltk.corpus import stopwords


def get_posPath(num):
    dirr = '/Users/Weiwei/Desktop/SentimentAnalysis/Corpus/pos/'
    posList = os.listdir('/Users/Weiwei/Desktop/SentimentAnalysis/Corpus/pos/')

    del posList[0]
    filepath = dirr+posList[num]
    return filepath


def get_negPath(num):
    dirr = '/Users/Weiwei/Desktop/SentimentAnalysis/Corpus/neg/'
    negList = os.listdir('/Users/Weiwei/Desktop/SentimentAnalysis/Corpus/neg/')
    del negList[0]
    filepath = dirr+negList[num]
    return filepath


def get_doc(fpath):
    txtfile = open(fpath, 'r')
    txt = txtfile.readlines()
    doc = ''.join(txt)
    return doc








def token_File(doc):
    sentences = nltk.sent_tokenize(doc)  # sentence-based separation(can enhance the efficiency while word-based separation)
    word_tokens = [nltk.word_tokenize(sent) for sent in sentences]  # word-based separation
    return word_tokens




def tokenByWord(doc):
    word_tokens = nltk.word_tokenize(doc)
    return word_tokens




def word_features(word):
    dict ={}
    for i in xrange(0,len(word)):
        dict[unicode(word[i])] = 'True'
    return dict


def regularExpression(doc):
    doc = re.sub(r'[^a-zA-Z0-9 ]', '', doc)
    return doc

def word_join(doc):
    joined = ''.join(doc)
    return joined
'''
def process_data(doc):
  tokens = ' '.join(doc)
  tokens = re.sub(r'[^a-zA-Z0-9 ]','',tokens)
  tokens = nltk.word_tokenize(tokens)
  stopwords = nltk.corpus.stopwords.words('english')
  tokens = [token.lower() for token in tokens if token.lower() not in stopwords]

  return tokens
'''