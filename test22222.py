import nltk
from nltk.corpus import stopwords
import re
import nltk.classify.scikitlearn
from sklearn import svm
import time



def predict(sentence, vocab, clf):
    if clf.predict([transform_sentence(sentence, vocab)]) == '0': print 'negative'
    else: print 'positive'

def transform_sentence(sentence, vocab):
    tokens = [x.lower() for x in nltk.word_tokenize(sentence) if not x in stops]
    fdist = nltk.FreqDist(tokens)
    features = [fdist[x] for x in vocab]
    return features

stops = stopwords.words('english')


data_file = open('/Users/Weiwei/Desktop/testdata.txt', 'r')
sentences = []
scores = []
for line in data_file:
    sentence, score = line.split("\t",1)
    sentences.append(sentence.decode('utf-8'))
    scores.append(score.strip())

#print sentences
#print scores

text =''.join(sentences)
text  = re.sub(r'[^a-zA-Z0-9 ]','',text)
tokens = nltk.word_tokenize(text)
tokens_filtered = [x.lower() for x in tokens if not x in stops]

freq_dist = nltk.FreqDist(tokens_filtered)

vocab = freq_dist.keys()


t0 = time.clock()
print t0
X = [transform_sentence(x, vocab) for x in sentences]


clf = svm.SVC(kernel = 'linear', C=1.0)
clf.fit(X,scores)

t1 = time.clock()
print t1
t = t1 - t0

print t


predict("It was fun!", vocab, clf)
predict("Horrible movie.", vocab, clf)
predict("It was only a small step up from their usual fiascos.", vocab, clf)
predict("I loved it only a little less than I love chocolate", vocab, clf)
predict("It's not my most favorite, but it's my least favorite.", vocab, clf)

