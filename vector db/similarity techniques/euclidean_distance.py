# import nlp from spacy
from math import sqrt, pow, exp
import spacy
# python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')

def squared_sum(x):
  """ return 3 rounded square rooted value """

  return round(sqrt(sum([a*a for a in x])),3)

def euclidean_distance(x,y):
  """ return euclidean distance between two lists """

  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))


embeddings = [nlp(sentence).vector for sentence in sentences]

distance = euclidean_distance(embeddings[0], embeddings[1])
print(distance)

# OUTPUT
1.8646982721454675

# distance can range from 0 to infinity
# use euler's constant for normalize

def distance_to_similarity(distance):
  return 1/exp(distance)

distance_to_similarity(distance)

# OUTPUT
0.8450570465624478