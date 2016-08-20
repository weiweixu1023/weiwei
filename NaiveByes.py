import tools
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util
import collections


negfilepath = []
posfilepath = []
pos_features = []
neg_features = []


for i in xrange(0,1000):                                 # get all the reviews' path
    posfilepath.append((tools.get_posPath(i)))           #
    negfilepath.append((tools.get_negPath(i)))           # and save them into two list(posfilepath and negfilepath)


for path in posfilepath:
    pos_features += [(tools.word_features(tools.tokenByWord(tools.get_doc(path))),'pos')]       #label the review
                                                                                                # and record all features
                                                                                                # into dictionary as the key,
for path in negfilepath:                                                                        # the value is True
    neg_features += [(tools.word_features(tools.tokenByWord(tools.get_doc(path))), 'neg')]




training_size = 800
test_size = 1000 - training_size

training_data = neg_features[0:training_size]+pos_features[0:training_size]  #data using to train classifier
test_data = neg_features[training_size:1000]+pos_features[training_size:1000]  #data using to test classifier


print 'There are 2000 reviews, 1000 of them are positive, the other 1000 are negative. We use ',2*training_size,' reviews' \
      ' (',training_size,' positive, ',training_size,' negative) to train classifier, and test ' \
      'the classifier with other ',2000-2*training_size,' reviews. )'


classifier = NaiveBayesClassifier.train(training_data)   #train classifier with training data
reference = collections.defaultdict(set)
testResult = collections.defaultdict(set)

for i, (feats, label) in enumerate(test_data):
    reference[label].add(i)                             # record the correct label of test data
    observed = classifier.classify(feats)               # calssify the test data according to classifier
    testResult[observed].add(i)                         # record the classification results

print 'accuracy:', nltk.classify.util.accuracy(classifier, test_data)  #calculating the accuracy

classifier.show_most_informative_features(10)   #display top 10 the most informative features



posP = nltk.precision(reference['pos'], testResult['pos'])  #calculating the positive precision as posP
posR = nltk.recall(reference['pos'], testResult['pos'])     #calculating the positive recall as posR
negP = nltk.precision(reference['neg'], testResult['neg'])  #calculating the negative precision as posP
negR = nltk.recall(reference['neg'], testResult['neg'])     #calculating the negative recall as posR

posF=2*posP*posR/(posP+posR)
negF=2*negP*negR/(negP+negR)


print 'positivePrecision',posP
print 'positiveRecall', posR
print 'positiveF-measure ',posF


print 'negativePrecision' , negP
print 'negativeRecall' , negR
print 'negativeF-measure', negF
