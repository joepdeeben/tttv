import nltk, re, pprint
from nltk import word_tokenize
from nltk import pos_tag
from nltk import CFG
from nltk.parse.generate import generate
import random

grammar = CFG.fromstring("""
S -> VP | VP CC S | PRP 
VP -> VB NP | VB NP PP | VB NP CC VP | VB NP PP CC VP
NP -> DT JJ NNS | DT MD NNS | DT JJ NNS CC NP | DT NNS CC NP | DT CD NNS | DT JJS NNS | DT JJR NNP
PP -> IN NP | RB

DT -> 'the' | 'a' | 'all' | 'every' | 'this'|
NNS -> 'bean' | 'food' | 'garlic' | 'table' | 'salt' | 'circle' | 'oil' | 'pan' | 'onion' | 'potato' | 'swede' | 'carrot' | 'parsnip' | 'water' | 'rice' | 'coriander' | 'tomato' | 'chicken' | 'semolina' | 'khoya' | 'dough' | 'ball' | 'mango' | 'sweetener' | 'yogurt' | 'container' | 'sorbet' | 'base' | 'mixture' | 'fork' | 'bowl' | 'pinch' | 'salt' | 'flour' | 'custard' | 'galen' | 'milk' | 'lemon' | 'whey' | 'cheese' | 'cloth' | 'ginger' | 'nut' | 'rice' | 'sugar' | 'cardamom' | 'pistachio' | 'almond'
IN ->  'in' | 'of' | 'for' | 'so' | 'until' | 'after' | 'that' | 'into' | 'while' | 'with' | 'Once' | 'on' | 'over' | 'After' | 'as' | 'For' | 'till' | 'before' | 'As' | 'from' | 'under' 
VB -> 'leave' | 'add' | 'make' | 'fry' | 'mix' | 'heat' | 'simmer' | 'stir' | 'cook' | 'boil' | 'reduce' | 'check' | 'blend' | 'freeze' | 'beat' | 'scoop' | 'fill' | 'bring' | 'put' | 'separate' | 'beat' | 'pour' | 'turn' | 'wrap' | 'rinse' | 'squeeze' | 'transfer'
PRP -> 'it' | 'we' | 'you' | 'them' | 'they' 
CD -> '24 hours' | 'one' | '10' | '900ml' | '15' | '2' | '20' | '5' | '1' | 'two' | '4' | '30' | '40' | 'three'
MD -> 'can' | 'will' | 'may'
RB -> 'then' | 'often' | 'together' | 'thoroughly' | 'carefully' | 'very' | 'Immediately' | 'still' | 'occasionally' | 'gradually' | 'gently' | 'well' | 'only' | 'enough' | 'up' | 'again' | 'also' | 'Then' | 'quickly'
CC -> 'and'| 'or'| 'but'
RP -> 'up'| 'off'| 'out'
JJ -> 'soft'| 'mashed' | 'golden'| 'frying'| 'Next'| 'other'| 'same'| 'large'| 'chopped'| 'mixed'| 'deep'| 'little'| 'light'| 'green' | 'full'| 'homogeneous'| 'small'| 'delicate'| 'brown'| 'freezer-proof'| 'liquid'| 'firm'| 'fresh'| 'white'| 'hot'| 'heavy' | 'cold'| 'excess'| 'wrapped'| 'shallow'| 'top'| 'low'| 'mushy'| 'open'| 'creamy' | 'few' | 'necessary'
PDT -> 'all'
NNP -> 'Fry' | 'Heat' | 'Cover' | 'Serve' | 'Place' | 'Add' | 'Cook' | 'Mix' | 'Keep' | 'Saffron' | 'Sugar' | 'Blend' | 'smooth' | 'Pour' | 'Halph' | 'Fill' | 'Stir' | 'Seperate' | 'Tip' | 'Return'
JJS -> 'most'
JJR -> 'more' | 'lower'
( -> '('
) -> ')'
""")



# Extracts all the words from a certain category and puts them into a list
def extract_word_categories(grammar, symbol):
    word_categories = {}

    for production in grammar.productions():
        if production.lhs().symbol() in word_categories:
            word_categories[production.lhs().symbol()].extend(production.rhs())
        else:
            word_categories[production.lhs().symbol()] = list(production.rhs())

    return word_categories.get(symbol, [])


# All variables turned into lists
dt_list = extract_word_categories(grammar, 'DT')
nns_list = extract_word_categories(grammar, 'NNS')
in_list = extract_word_categories(grammar, 'IN')
vb_list = extract_word_categories(grammar, 'VB')
cc_list = extract_word_categories(grammar, 'CC')
rp_list = extract_word_categories(grammar, 'RP')
jj_list = extract_word_categories(grammar, 'JJ')
prp_list = extract_word_categories(grammar, 'PRP')
cd_list = extract_word_categories(grammar, 'CD')
md_list = extract_word_categories(grammar, 'MD')
rb_list = extract_word_categories(grammar, 'RB')
nnp_list = extract_word_categories(grammar, 'NNP')
jjs_list = extract_word_categories(grammar, 'JJS')
jjr_list = extract_word_categories(grammar, 'JJR')

# Generates a random NP, using one of the rules, based on a random number
def generate_np(dt, nns, jj, jjs, jjr):
    random_number = random.randint(1, 7)
    if random_number == 1:
        return random.choice(dt) + ' ' + random.choice(jj) + ' ' + random.choice(nns)
    elif random_number == 2:
        return random.choice(dt) + ' ' + random.choice(md_list) + ' ' + random.choice(nns)
    elif random_number == 3:
        return random.choice(dt) + ' ' + random.choice(jj) + ' ' + random.choice(nns) + ' ' + random.choice(cc_list) + ' ' + generate_np(dt, nns, jj, jjs, jjr)
    elif random_number == 4:
        return random.choice(dt) + ' ' + random.choice(nns) + ' ' + random.choice(cc_list) + ' ' + generate_np(dt, nns, jj, jjs, jjr)
    elif random_number == 5:
        return random.choice(dt) + ' ' + random.choice(cd_list) + ' ' + random.choice(nns)
    elif random_number == 6:
        return random.choice(dt) + ' ' + random.choice(jjs) + ' ' + random.choice(nns)
    elif random_number == 7:
        return random.choice(dt) + ' ' + random.choice(jjr_list) + ' ' + random.choice(nnp_list)

# Generates a random PP or an adverb
def generate_pp(in_words, np):
    random_number = random.randint(1, 2)
    if random_number == 1:
        return random.choice(in_words) + ' ' + np
    elif random_number == 2:
        return random.choice(rb_list)

# Generates a random VP, using one of the rules, based on a random number
def generate_vp(vb, np, pp):
    random_number = random.randint(1, 4)
    if random_number == 1:
        return random.choice(vb) + ' ' + np
    elif random_number == 2:
        return random.choice(vb) + ' ' + np + ' ' + pp
    elif random_number == 3:
        return random.choice(vb) + ' ' + np + ' ' + random.choice(cc_list) + ' ' + generate_vp(vb, np, pp)
    elif random_number == 4:
        return random.choice(vb) + ' ' + random.choice(prp_list) + ' ' + random.choice(rp_list)

def generate_sentences(amount):
    sentences = []
    for _ in range(amount):
        sentence = generate_vp(vb_list, generate_np(dt_list, nns_list, jj_list, jjs_list, jjr_list), generate_pp(in_list, generate_np(dt_list, nns_list, jj_list, jjs_list, jjr_list)))
        sentences.append(sentence)
    return sentences

sentences = generate_sentences(50)
for sentence in sentences:
    print(sentence)