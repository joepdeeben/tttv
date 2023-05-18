import nltk, re, pprint
from nltk import word_tokenize
from nltk import pos_tag
from nltk import CFG
from nltk.parse.generate import generate
import random

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
RP -> 'up'| 'off'| 'out'
N -> 'bruh' | 'test'

""")

# Extracts all the words from a certain category and puts it into a list
def extract_word_categories(grammar, symbol):
    word_categories = {}

    for production in grammar.productions():
        if production.lhs().symbol() in word_categories:
            word_categories[production.lhs().symbol()].extend(production.rhs())
        else:
            word_categories[production.lhs().symbol()] = list(production.rhs())

    return word_categories.get(symbol, [])

# All variables turned into list
dt_list = extract_word_categories(grammar, 'DT')
nns_list = extract_word_categories(grammar, 'NNS')
in_list = extract_word_categories(grammar, 'IN')
vb_list = extract_word_categories(grammar, 'VB')
cc_list = extract_word_categories(grammar, 'CC')
rp_list = extract_word_categories(grammar, 'RP')
n_list = extract_word_categories(grammar, 'N')
SPACE = ' '

# Generates a random NP, using one of the two rules, based on  a random number
def generate_np(dt, nns, n):
    random_number = random.randint(1, 2)
    if random_number == 1:
        return random.choice(dt) + SPACE + random.choice(n)
    else:
        return random.choice(dt) + SPACE + random.choice(nns)

# Generates a random PP
def generate_pp(in_words, np):
    return random.choice(in_words) + SPACE + np

# Generates a random VP
def generate_vp(vb, np, pp):
    random_number = random.randint(5, 6)
    if random_number == 5:
        return random.choice(vb) + SPACE + np + SPACE + pp
    else:
        return random.choice(vb) + SPACE + np

def generate_sentences(amount):
    sentences = []
    for i in range(amount):
        sentence = generate_vp(vb_list, generate_pp(in_list, generate_np(dt_list, nns_list, n_list)), generate_np(dt_list, nns_list, n_list))
        sentences.append(sentence)
    return sentences

sentences = generate_sentences(200)
for sentence in sentences:
    print(sentence)