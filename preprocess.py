import nltk, re, pprint
from nltk import word_tokenize
from nltk import pos_tag
from nltk import CFG
from nltk.parse.generate import generate

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
    print(category, '->')
    for word in tagged:
        if (word[1] == category) and word not in words_checked:
            print(f"'{word[0]}'|" )
            words_checked.append(word)

grammar = CFG.fromstring("""
S -> VP
VP -> VB NP | VB NP PP
NP -> DT N | DT NNS
PP -> IN NP

DT -> 'the' | 'a' | 'all' | 'every' | 'The' | 'this'|
NNS -> 'beans' | 'bowl' | 'food' | 'garlic' | 'table' | 'salt' | 'circle' | 'oil' | 'pan' | 'onion' | 'potatoes' | 'swede' | 'carrots' | 'parsnips' | 'water' | 'rice' | 'coriander' | 'tomatoes' | 'chicken' | 'semolina' | 'khoya' | 'dough' | 'balls' | 'mango' | 'sweetener' | 'yogurt' | 'container' | 'sorbet' | 'base' | 'mixture' | 'fork' | 'bowls' | 'pinch' | 'salt' | 'flour' | 'custard' | 'galen' | 'milk' | 'lemon' | 'whey' | 'cheese' | 'cloth' | 'ginger' | 'nuts' | 'rice' | 'sugar' | 'cardamom' | 'pistachios' | 'almonds'
IN ->  'in' | 'of' | 'for' | 'so' | 'until' | 'after' | 'that' | 'into' | 'while' | 'with' | 'Once' | 'on' | 'if' | 'over' | 'After' | 'as' | 'For' | 'till' | 'before' | 'tard' | 'Boil' | 'As' | 'from' | 'under' | 'about' | 'If'
VB -> 'leave' | 'add' | 'make' | 'fry' | 'mix' | 'heat' | 'simmer' | 'stir' | 'cook' | 'boil' | 'reduce' | 'check' | 'blend' | 'freeze' | 'beat' | 'scoop' | 'fill' | 'bring' | 'put' | 'separate' | 'beat' | 'pour' | 'turn' | 'wrap' | 'rinse' | 'squeeze' | 'transfer'
CC -> 'and'| 'or'| 'but'
RP -> 'up'| 'off'| 'out'|

""")

length = int(input("how deep should the generation be? \n"))
for sentence in generate(grammar, depth=length):
    print(' '.join(sentence))

