import nltk, re, pprint
from nltk import word_tokenize
from nltk import pos_tag
from nltk import CFG
from nltk.parse.generate import generate
import random

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
NNS -> 'bean' | 'food' | 'garlic' | 'table' | 'salt' | 'circle' | 'oil' | 'pan' | 'onion' | 'potato' | 'swede' | 'carrot' | 'parsnip' | 'water' | 'rice' | 'coriander' | 'tomato' | 'chicken' | 'semolina' | 'khoya' | 'dough' | 'ball' | 'mango' | 'sweetener' | 'yogurt' | 'container' | 'sorbet' | 'base' | 'mixture' | 'fork' | 'bowl' | 'pinch' | 'salt' | 'flour' | 'custard' | 'galen' | 'milk' | 'lemon' | 'whey' | 'cheese' | 'cloth' | 'ginger' | 'nut' | 'rice' | 'sugar' | 'cardamom' | 'pistachio' | 'almond'
IN ->  'in' | 'of' | 'for' | 'so' | 'until' | 'after' | 'that' | 'into' | 'while' | 'with' | 'Once' | 'on' | 'if' | 'over' | 'After' | 'as' | 'For' | 'till' | 'before' | 'tard' | 'Boil' | 'As' | 'from' | 'under' | 'about' | 'If'
VB -> 'leave' | 'add' | 'make' | 'fry' | 'mix' | 'heat' | 'simmer' | 'stir' | 'cook' | 'boil' | 'reduce' | 'check' | 'blend' | 'freeze' | 'beat' | 'scoop' | 'fill' | 'bring' | 'put' | 'separate' | 'beat' | 'pour' | 'turn' | 'wrap' | 'rinse' | 'squeeze' | 'transfer'
CC -> 'and'| 'or'| 'but'
RP -> 'up'| 'off'| 'out'|

""")

def generate_sentence(grammar, symbol):
    if isinstance(symbol, str):
        productions = grammar.productions(lhs=symbol)
        production = random.choice(productions)
        return ' '.join(generate_sentence(grammar, sym) for sym in production.rhs())
    else:
        return 'bruh'

# Generate a sentence using the CFG
generated_sentence = generate_sentence(grammar, grammar.start())
print(generated_sentence)