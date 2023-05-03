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
S -> NP VP
VP -> VB NN | VB NN PP | VB NNS | VB NNS PP
PP -> IN NN | IN NNS
NP -> IN DT NN VB | IN DT NN VB PP

PRP -> 'it'| 'we'|'you'|'them'|'they'|'They'|'You'

IN -> 'in'|'of'|'for'|'so'|'until'|'after'|'that'|'into'|'while'|'with'|'Once'|'on'|'if'|'over'|'After'|'as'|'For'|'till'|'before'|'tard'|'Boil'|'As'|'from'|'under'|'about'|'If'

VB -> 'Put'| 'be'| 'Make'| 'shape'| 'serve'| 'Place'| 'boil'| 'reduce'| 'Add'| 'Boil'| 'skin'| 'cut'| 'Chop'| 'cook'| 'check'| 'make'| 'separate'| 'let'| 'simmer'| 'Take'| 'soak'| 'set'| 'Beat'| 'return'| 'break'| 'Transfer'| 'fridge'| 'Buy'| 'burn'| 'stir'| 'start'| 'turn'| 'muslin'| 'Wrap'| 'take'| 'prevent'| 'scoop'| 'add'| 'cover'| 'cool'| 'have'| 'dip'| 'finish'| 'Repeat'| 'need'| 'do'| 'pan'

DT -> 'the'|'a'|'all'|'every'|'every'|'this'

NNS -> 'beans'| 'ingredients'| 'carrots'| 'parsnips'| 'minutes'| 'cubes'| 'onions'| 'tomatoes'| 'khoya'| 'balls'| 'cups'| 'hours'| 'sides'| 'â€'| 'crystals'| 'mintures'| 'slices'| 'spoons'| 'boils'| 'cloth'| 'nuts'| 'cristals'| 'flavour'| 'pools'| 'seeds'| 'times'| 'pieces'| 'almonds'| 'pistachios'| 'yolks'| 'fingers'| 'tbsp'| 'layers'| 'mins'| 'batches'

NN -> 'bowl'| 'water'| 'food'| 'processor'| 'bit'| 'powder'| 'spoon'| 'salt'| 'circle'| 'shape'| 'brown'| 'medium'| 'size'| 'mix'| 'oil'| 'frying'| 'pan'| 'mixture'| 'pakoras'| 'batch'| 'onion'| 'potatos'| 'swede'| 'heat'| 'simmer'| 'min'| 'curry'| 'chat'| 'masala'| 'stir'| 'cook'| 'rice'| 'liquid'| 'coriander'| 'cup'| 'fry'| 'fryer'| 'skin'| 'cooked'| 'garlic'| 'wait'| 'chopped'| 'chicken'| 'garam'| 'chilli'| 'corriander'| 'semolina'| 'dough'| 'hour'| '.This'| 'light'| 'spongy'| 'round'| 'image'| 'sugar'| 'syrup'| 'ghee'| 'color'| 'mango'| 'flesh'| 'pro-'| 'cessor'| 'sweetener'| 'yogurt'| 'container'| 'Freeze'| 'base'| 'sorbet'| 'center'| 'fork'| 'freezer'| 'ince'| 'bowls'| 'flour'| 'pinch'| 'kg'| 'plain'| 'bt'| 'cake'| 'half'| 'galen'| 'milk'| 'custard'| 'lemon'| 'juice'| 'bottomed'| 'curd'| 'whey'| 'fat'| 'strainer'| 'line'| 'cloth'| 'muslin'| 'rinse'| 'squeeze'| 'process'| 'sourness'| 'press'| 'paneer'| 'ginger'| 'proof'| 'freeze'| 'ice'| 'cream'| 'garnish'| 'sauce'| 'bring'| 'boil'| 'till'| 'wate'| 'cardamom'| 'kheer'| 'ou'| 'warm'| 'cold'| 'serve'| 'egg'| 'mascarpone'| 'beat'| 'coffee'| 'cacao'| 'batter'| 'top'| 'pone'| 'lady'| 'finger'| 'tbsp'| 'dust'| 'chocolate'| 'ground'| 'potato'| 'stock'| 'tender'| 'der'| 'taste'| 'reheat'


""")

length = int(input("how deep should the generation be? \n"))
for sentence in generate(grammar, depth=length):
    print(' '.join(sentence))

print(categories)
print(tagged)
