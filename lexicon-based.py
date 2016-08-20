##########################################
'''
'''

############################################

import nltk
from nltk.corpus import sentiwordnet as swn

import tools

review = ['negative',4]  #(positive,negative)(0-999) choose a review


if review [0] == 'positive': fpath = tools.get_posPath(review[1])
if review [0] == 'negative': fpath = tools.get_negPath(review[1])



txtfile = open(fpath,'r')
txt = txtfile.readlines()
doc = ''.join(txt)

sentences = nltk.sent_tokenize(doc)   #sentence-based separation(can enhance the efficiency while word-based separation)


word_tokens = [nltk.word_tokenize(sent) for sent in sentences]  #word-based separation
print word_tokens

taglist = []
for word_token in word_tokens:
    taglist.append(nltk.pos_tag(word_token))  # part of speech marked
word_lemma = nltk.WordNetLemmatizer()
score_list = []

for word_index, tagged in enumerate(taglist):  #enumerate: list
    score_list.append([])  #save the polarity score
    for tempindex, word_tag in enumerate(tagged):
        newtag = ''
        lemmatized = word_lemma.lemmatize(word_tag[0]) #lemmatization
        if word_tag[1].startswith('NN'):
            newtag = 'n'            #NN,NNP,NNPS,NNS,
        elif word_tag[1].startswith('JJ'):
            newtag = 'a'            #JJ,JJR,JJS
        elif word_tag[1].startswith('V'):
            newtag = 'v'            #VB,VBD,VBG,VBN,VBP,VBZ
        elif word_tag[1].startswith('R'):
            newtag = 'r'            #RB,RBR,RBS,RP
        else:
            newtag = ''

        if (newtag != ''):
            synsets = list(swn.senti_synsets(lemmatized, newtag))  #ignore useless words, which won't affect result,
                                                                    # and get all sense and synonyms of this word.
            score = 0
            if (len(synsets) != 0): #have synsets
                for syn in synsets:
                    score += (syn.pos_score() - syn.neg_score()) #calculate the sentiment score of this word
                    denominator = len(synsets) #in order to avoid denominator = 0,len +1
                score_list[word_index].append(score / denominator)    # Getting average of all possible sentiments
sentence_sentiment = []


for score_sent in score_list:
    sentence_sentiment.append(sum([word_score for word_score in score_sent]) /( len(score_sent)+1))  #calculate the sentiment score of a sentence
print("Sentiment for the review for: " + fpath)
sentiment_sum = sum(sentence_sentiment)
if sentiment_sum >0:
    judge = 'Positive'             #score > 0 positive
elif sentiment_sum < 0:
    judge = 'Negative'             #score < 0 negative
else :  #sentence_sentiment == 0
    judge = 'Neutral'              #score = 0 neutral
print'Score :',sentiment_sum
print 'Polarity :',judge
print  'Content : ',doc
