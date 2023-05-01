import nltk, re, pprint
from nltk import word_tokenize
from nltk import pos_tag

f = open('corpus.txt', 'r')
raw = f.readlines()
rawstr = ''


for sentence in raw:
    rawstr += sentence.strip('\n') + " "

tokenized = word_tokenize(rawstr)
tagged = pos_tag(tokenized)

#creëer de verschillende categorieën
categories = []
for word in tagged:
    if word[1] in categories:
        pass
    else:
        categories.append(word[1])

#woorden categoriseren, dit is puur voor ons om een overzicht te geven
words_checked = []
for category in categories:
    print(category)
    for word in tagged:
        if (word[1] == category) and word not in words_checked:
            print(word)
            words_checked.append(word)



print(categories)
print(tagged)
