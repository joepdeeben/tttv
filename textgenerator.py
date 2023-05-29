import nltk, re, pprint
from nltk import word_tokenize
from nltk import pos_tag
from nltk import CFG
from nltk.parse.generate import generate
import random

grammar = CFG.fromstring("""
S -> VP RB PRP VP
S -> VP
S -> VP CC VP
S -> VP NP
S -> NP VP
S -> NP VP CC VP


VP -> VB NP PP | VB VBD PP | VB IN JJ | VB IN CD | VB PRP IN | MD VP | VB JJ | VP CC VP
NP -> DT NNS | DT NN | NN | NNS | DT NNS PP | JJ NN | JJ NNS | CD NNS | PRP | NP PP | PDT DT NNS | NP CC NP| DT JJ NN NN
PP -> IN NP 




DT -> 'the' | 'a' | 'all' | 'Every' | 'every' | 'The' | 'This' | 'this'
VB ->  'Return' | 'reheat' | 'Tip' | 'Seperate' | 'Stir' | 'Fill' | 'Pour' | 'Blend' | 'Keep' | 'Mix' | 'Cook' | 'Serve' | 'Cover' | 'Heat' | 'Fry' | 'Put'| 'be'| 'Make'| 'shape'| 'serve'| 'Place'| 'boil'| 'reduce'| 'Add'| 'Boil'| 'skin'| 'cut'| 'Chop'| 'cook'| 'check'| 'make'| 'separate'| 'let'| 'simmer'| 'Take'| 'soak'| 'set'| 'Beat'| 'return'| 'break'| 'Transfer'| 'fridge'| 'Buy'| 'burn'| 'stir'| 'start'| 'turn'| 'muslin'| 'Wrap'| 'take'| 'prevent'| 'scoop'| 'add'| 'cover'| 'cool'| 'have'| 'dip'| 'finish'| 'Repeat'| 'need'| 'do'| 'pan'
NNS -> 'gallon' | 'cheese' | 'beans'| 'hours'| 'ingredients'| 'carrots'| 'parsnips'| 'minutes'| 'cubes'| 'onions'| 'tomatoes'| 'khoya'| 'balls'| 'cups'| 'sides'| 'crystals'| 'mintures'| 'slices'| 'spoons'| 'boils'| 'cloth'| 'nuts'| 'cristals'| 'flavour'| 'pools'| 'seeds'| 'times'| 'pieces'| 'almonds'| 'pistachios'| 'yolks'| 'fingers'| 'tbsp'| 'layers'| 'mins'| 'batches'
IN -> 'in'| 'of'| 'for'| 'so'| 'until'| 'after'| 'that'| 'into'| 'while'| 'with'| 'Once'| 'on'| 'if'| 'over'| 'After'| 'as'| 'For'| 'till'| 'before'| 'tard'| 'Boil'| 'As'| 'from'| 'under'| 'about'| 'If'
NN -> 'Sugar' | 'Cardamom'| 'Saffron' | 'Gulab' | 'Jamun' | 'bowl'| 'water'| 'food'| 'processor'| 'bit'| 'powder'| 'spoon'| 'salt'| 'circle'| 'shape'| 'brown'| 'medium'| 'size'| 'mix'| 'oil'| 'frying'| 'pan'| 'mixture'| 'pakoras'| 'batch'| 'onion'| 'potatos'| 'swede'| 'heat'| 'simmer'| 'min'| 'curry'| 'chat'| 'masala'| 'stir'| 'cook'| 'rice'| 'liquid'| 'coriander'| 'cup'| 'fry'| 'fryer'| 'skin'| 'cooked'| 'garlic'| 'wait'| 'chopped'| 'chicken'| 'garam'| 'chilli'| 'corriander'| 'semolina'| 'dough'| 'hour'| 'light'| 'spongy'| 'round'| 'image'| 'sugar'| 'syrup'| 'ghee'| 'color'| 'mango'| 'flesh'| 'processor'| 'sweetener'| 'yogurt'| 'container'| 'Freeze'| 'base'| 'sorbet'| 'center'| 'fork'| 'freezer'| 'ince'| 'bowls'| 'flour'| 'pinch'| 'kg'| 'plain'| 'bt'| 'cake'| 'half'| 'galen'| 'milk'| 'custard'| 'lemon'| 'juice'| 'bottomed'| 'curd'| 'whey'| 'fat'| 'strainer'| 'line'| 'cloth'| 'muslin'| 'rinse'| 'squeeze'| 'process'| 'sourness'| 'press'| 'paneer'| 'ginger'| 'proof'| 'freeze'| 'ice'| 'cream'| 'garnish'| 'sauce'| 'bring'| 'boil'| 'till'| 'wate'| 'cardamom'| 'kheer' | 'warm'| 'cold'| 'serve'| 'egg'| 'mascarpone'| 'beat'| 'coffee'| 'cacao'| 'batter'| 'top'| 'pone'| 'lady'| 'finger'| 'tbsp'| 'dust'| 'chocolate'| 'ground'| 'potato'| 'stock'| 'tender'|  'taste'
COMMA -> ','
CC -> 'and'| 'or'| 'but' | '/' | '–' | 'so' 
VBP -> 'leave'| 'become'| 'add'| 'want'| 'are'| 'do'| 'make'| 'attains'| 'have'| 'mix'| 'drain'
PRP -> 'it'| 'we'| 'you'| 'them'| 'they'| 'They'| 'You'
CD -> '24'| 'one'| '10'| '900ml'| '15'| '2'| '20'| '5'| '1'| 'two'| '4'| '30'| '40'| 'three' | '3' | '-5'
MD -> 'can'| 'will'| 'may'
JJ -> 'soft'| 'mashed'| 'garlic'| 'table'| 'colour'| 'golden'| 'frying'| 'Next'| 'other'| 'same'| 'ready'| 'smooth' | 'large'| 'further'| 'chopped'| '¼'| 'mixed'| 'deep'| 'little'| 'light'| 'green'| '15-20'| 'bowl'| 'full'| 'homogeneous'| 'small'| 'delicate'| 'brown'| 'freezer-proof'| 'liquid' | 'firm' | 'fresh'| 'white'| 'hot'| 'heavy'| 'sure'| 'cold'| 'excess'| 'wrapped'| 'shallow'| 'top'| 'low'| 'mushy'| 'add'| 'open'| 'creamy'| 'cacao-mascarpone'| 'coffee-mascarpone'| 'few'| 'necessary'
VBZ -> 'becomes'| 'is'| 'has'| 'helps'| 'Sprinkles'| 'comes'| 'takes'
PDT -> 'all'
VBG -> 'heating'| 'boiling'| 'bring'| 'stirring'| 'making'| 'beating'| 'scooping'| 'separating'| 'using'| 'stiring'| 'forming'| 'remaining'
TO -> 'to'| 'To'
VBD -> 'put'| 'sprinkled'| 'garnished'| 'chopped'| 'boiled'| 'reduced'| 'broke'| 'blitz' | 'smooth'
RB -> 'then'| 'often'| 'together'| 'brown'| 'thoroughly'| 'carefully'| 'very'| 'Immediately'| 'still'| 'aside'| 'occasionally'| 'not'| 'gradually'| 'gently'| 'well'| 'put'| 'only'| 'milk'| 'enough'| 'up'| 'again'| 'also'| 'Then'| 'quickly'
VBN -> 'put'| 'cooked'| 'been'| 'absorbed'| 'dried'| 'stir'| 'shown'| 'started'| 'separated'| 'done'| 'kheer'| 'softened'
PRPS -> 'their'| 'its' | 'You'
WRB -> 'When'
RP -> 'up'| 'off'| 'out'
JJS -> 'most'
JJR -> 'more'| 'lower'
OPEN -> '('
CLOSE -> ')'
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
nns_list = extract_word_categories(grammar, 'NNS')
nn_list = extract_word_categories(grammar, 'NN')
dt_list = extract_word_categories(grammar, 'DT')
jj_list = extract_word_categories(grammar, 'JJ')
vb_list = extract_word_categories(grammar, 'VB')
in_list = extract_word_categories(grammar, 'IN')
cc_list = extract_word_categories(grammar, 'CC')
prp_list = extract_word_categories(grammar, 'PRP')
cd_list = extract_word_categories(grammar, 'CD')

# Generates a random NP, using one of the rules, based on a random number
def generate_np():
    random_number = random.randint(1, 7)
    if random_number == 1:
        return random.choice(dt_list) + ' ' + random.choice(nns_list)
    elif random_number == 2:
        return random.choice(dt_list) + ' ' + random.choice(nn_list)
    elif random_number == 3:
        return random.choice(nn_list)
    elif random_number == 4:
        return random.choice(nns_list)
    elif random_number == 5:
        return random.choice(dt_list) + ' ' + random.choice(nns_list) + ' ' + generate_pp()
    elif random_number == 6:
        return random.choice(jj_list) + ' ' + random.choice(nn_list)
    elif random_number == 7:
        return random.choice(jj_list) + ' ' + random.choice(nns_list)

# Generates a random PP
def generate_pp():
    return random.choice(in_list) + ' ' + generate_np()

# Generates a random VP, using one of the rules, based on a random number
def generate_vp():
    random_number = random.randint(1, 6)
    if random_number == 1:
        return random.choice(vb_list) + ' ' + generate_np() + ' ' + generate_pp()
    elif random_number == 2:
        return random.choice(vb_list) + ' ' + generate_np() + ' ' + generate_pp()
    elif random_number == 3:
        return random.choice(vb_list) + ' ' + random.choice(in_list) + ' ' + random.choice(jj_list)
    elif random_number == 4:
        return random.choice(vb_list) + ' ' + random.choice(in_list) + ' ' + random.choice(cd_list)
    elif random_number == 5:
        return random.choice(vb_list) + ' ' + random.choice(prp_list) + ' ' + random.choice(in_list)
    elif random_number == 6:
        return random.choice(vb_list)

def generate_sentences(amount):
    sentences = []
    for _ in range(amount):
        sentence = generate_vp()
        sentences.append(sentence)
    return sentences

sentences = generate_sentences(50)
for sentence in sentences:
    print(sentence)