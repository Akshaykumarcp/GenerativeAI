from math import sqrt, pow, exp
import spacy
# python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')

embeddings = [nlp(sentence).vector for sentence in sentences]

def squared_sum(x):
  """ return 3 rounded square rooted value """

  return round(sqrt(sum([a*a for a in x])),3)

def cos_similarity(x,y):
  """ return cosine similarity between two lists """

  numerator = sum(a*b for a,b in zip(x,y))
  denominator = squared_sum(x)*squared_sum(y)
  return round(numerator/float(denominator),3)

cos_similarity(embeddings[0], embeddings[1])

# OUTPUT
0.891