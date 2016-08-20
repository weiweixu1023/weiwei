import nltk
from nltk.corpus import stopwords
import re
import nltk.classify.scikitlearn
from sklearn import svm
import tools




test_score = []


def predict(sentence, vocab, clf):
    print(clf.predict([transform_sentence(sentence, vocab)]))
'''

def predict(sentence, vocab, clf):
    polariy = (clf.predict([transform_sentence(sentence, vocab)]))
    return polariy
'''


def transform_sentence(sentence, vocab):
    tokens = [x.lower() for x in nltk.word_tokenize(sentence) if not x in stops]
    fdist = nltk.FreqDist(tokens)
    features = [fdist[x] for x in vocab]
    return features

stops = stopwords.words('english')


######################################################################################################################
posPath = []
negpath = []

sentences = []
testsentences = []
possentences = []
negsentences = []
combine_sentences = []

scores = []
testscores = []
posscores = []
negscores = []
combine_scores = []

train_proportion = 0.8 #(0-1)
polarity_setsize = 1000

for i in xrange(0,polarity_setsize):
    posPath.append(tools.get_posPath(i))
    negpath.append(tools.get_negPath(i))

# positive
for path in posPath:
    pos_file = open(path,'r')
    postext = ''.join(pos_file)
    possentences.append(postext.decode('utf-8'))
    posscores.append('1')

#negative
for path in  negpath:
    neg_file = open(path, 'r')
    negtext = ''.join(neg_file)
    negsentences.append(negtext.decode('utf-8'))
    negscores.append('0')

#combine
for i in xrange(0,polarity_setsize):
    combine_sentences.append(possentences[i])
    combine_sentences.append(negsentences[i])
    combine_scores.append(posscores[i])
    combine_scores.append(negscores[i])



#trainset
train_size = int(2*polarity_setsize*train_proportion)
sentences = combine_sentences[:train_size]
scores = combine_scores[:train_size]

#print sentences
#print len(sentences)

#testset
testsentences = combine_sentences[train_size:]
testscores = combine_scores[train_size:]

#print '\nTest'
#print testsentences
#print len(testsentences)
#print testscores
#print len(testscores)




######################################################################################################################

text =''.join(sentences)
text  = re.sub(r'[^a-zA-Z0-9 ]','',text)
tokens = nltk.word_tokenize(text)
tokens_filtered = [x.lower() for x in tokens if not x in stops]


freq_dist = nltk.FreqDist(tokens_filtered)

vocab = freq_dist.keys()

X = [transform_sentence(x, vocab) for x in sentences]


clf = svm.SVC(kernel = 'linear', C=1.0)
clf.fit(X,scores)


for text in testsentences:
    predict(text,vocab,clf)









'''
predict("It was fun!", vocab, clf)
predict("Horrible movie.", vocab, clf)
predict("It was only a small step up from their usual fiascos.", vocab, clf)
predict("I loved it only a little less than I love chocolate", vocab, clf)
predict("It's not my most favorite, but it's my least favorite.", vocab, clf)
'''