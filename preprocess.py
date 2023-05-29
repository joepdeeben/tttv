import nltk
from nltk import word_tokenize, CFG
from nltk.parse import BottomUpLeftCornerChartParser
from nltk import pos_tag
from nltk.tree import Tree

# Read the corpus
with open('corpus.txt', 'r') as f:
    raw = f.read()

# Split the raw text into sentences
sentences = raw.split('.')

# Define the noun phrase pattern
grammar = CFG.fromstring("""
S -> VP RB PRP VP
S -> VP
S -> VP CC VP
S -> VP NP
S -> NP VP
S -> NP VP CC VP


VP -> VB NP PP | VB VBD PP | VB IN JJ | VB IN CD | VB PRP IN | MD VP | VB JJ | VP CC VP | VB | VP CC NP
NP -> DT NNS | DT NN | NN | NNS | DT NNS PP | JJ NN | JJ NNS | CD NNS | PRP | NP PP | PDT DT NNS | NP CC NP| DT JJ NN NN 
PP -> IN NP 




DT -> 'the' | 'a' | 'all' | 'Every' | 'every' | 'The' | 'This' | 'this'
VB ->  'Return' | 'reheat' | 'Tip' | 'Seperate' | 'Stir' | 'Fill' | 'Pour' | 'Blend' | 'Keep' | 'Mix' | 'Cook' | 'Serve' | 'Cover' | 'Heat' | 'Fry' | 'Put'| 'be'| 'Make'| 'shape'| 'serve'| 'Place'| 'boil'| 'reduce'| 'Add'| 'Boil'| 'skin'| 'cut'| 'Chop'| 'cook'| 'check'| 'make'| 'separate'| 'let'| 'simmer'| 'Take'| 'soak'| 'set'| 'Beat'| 'return'| 'break'| 'Transfer'| 'fridge'| 'Buy'| 'burn'| 'stir'| 'start'| 'turn'| 'muslin'| 'Wrap'| 'take'| 'prevent'| 'scoop'| 'add'| 'cover'| 'cool'| 'have'| 'dip'| 'finish'| 'Repeat'| 'need'| 'do'| 'pan'
NNS -> 'gallon' | 'cheese' | 'beans'| 'hours'| 'ingredients'| 'carrots'| 'parsnips'| 'minutes'| 'cubes'| 'onions'| 'tomatoes'| 'khoya'| 'balls'| 'cups'| 'sides'| 'crystals'| 'mintures'| 'slices'| 'spoons'| 'boils'| 'cloth'| 'nuts'| 'cristals'| 'flavour'| 'pools'| 'seeds'| 'times'| 'pieces'| 'almonds'| 'pistachios'| 'yolks'| 'fingers'| 'tbsp'| 'layers'| 'mins'| 'batches'
IN -> 'in'| 'of'| 'for'| 'so'| 'until'| 'after'| 'that'| 'into'| 'while'| 'with'| 'Once'| 'on'| 'if'| 'over'| 'After'| 'as'| 'For'| 'till'| 'before'| 'tard'| 'Boil'| 'As'| 'from'| 'under'| 'about'| 'If'
NN -> 'mix' | 'Sugar' | 'Cardamom'| 'Saffron' | 'Gulab' | 'Jamun' | 'bowl'| 'water'| 'food'| 'processor'| 'bit'| 'powder'| 'spoon'| 'salt'| 'circle'| 'shape'| 'brown'| 'medium'| 'size'| 'mix'| 'oil'| 'frying'| 'pan'| 'mixture'| 'pakoras'| 'batch'| 'onion'| 'potatos'| 'swede'| 'heat'| 'simmer'| 'min'| 'curry'| 'chat'| 'masala'| 'stir'| 'cook'| 'rice'| 'liquid'| 'coriander'| 'cup'| 'fry'| 'fryer'| 'skin'| 'cooked'| 'garlic'| 'wait'| 'chopped'| 'chicken'| 'garam'| 'chilli'| 'corriander'| 'semolina'| 'dough'| 'hour'| 'light'| 'spongy'| 'round'| 'image'| 'sugar'| 'syrup'| 'ghee'| 'color'| 'mango'| 'flesh'| 'processor'| 'sweetener'| 'yogurt'| 'container'| 'Freeze'| 'base'| 'sorbet'| 'center'| 'fork'| 'freezer'| 'ince'| 'bowls'| 'flour'| 'pinch'| 'kg'| 'plain'| 'bt'| 'cake'| 'half'| 'galen'| 'milk'| 'custard'| 'lemon'| 'juice'| 'bottomed'| 'curd'| 'whey'| 'fat'| 'strainer'| 'line'| 'cloth'| 'muslin'| 'rinse'| 'squeeze'| 'process'| 'sourness'| 'press'| 'paneer'| 'ginger'| 'proof'| 'freeze'| 'ice'| 'cream'| 'garnish'| 'sauce'| 'bring'| 'boil'| 'till'| 'wate'| 'cardamom'| 'kheer' | 'warm'| 'cold'| 'serve'| 'egg'| 'mascarpone'| 'beat'| 'coffee'| 'cacao'| 'batter'| 'top'| 'pone'| 'lady'| 'finger'| 'tbsp'| 'dust'| 'chocolate'| 'ground'| 'potato'| 'stock'| 'tender'|  'taste'
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

i = 0
j = 0
# Initialize the parser with the grammar
parser = BottomUpLeftCornerChartParser(grammar)

# Iterate over each sentence in our corpus
for sentence in sentences:
    # Remove commas from the input sentence
    sentence_without_commas = sentence.replace(',', '')

    # Tokenize the input sentence
    tokenized_sentence = word_tokenize(sentence_without_commas)

    try:
        print(tokenized_sentence)
        # Check if a parse tree can be found
        parse_tree = next(parser.parse(tokenized_sentence))

        # Print the parse tree
        tree = Tree.fromstring(str(parse_tree))
        tree.pretty_print()

        # Translate the parse tree
        translation = ' '.join(node.label() for node in parse_tree if isinstance(node, nltk.Tree))
        
        
        # Print the result
        print("Sentence:", sentence.strip())
        print("Translation:", translation)
        i += 1
    except StopIteration:
        print("No parse tree found for sentence:", sentence.strip())
        j += 1
    print()
print('Goeie zinnen: ', + i)
print('Foute zinnen: ', + j)