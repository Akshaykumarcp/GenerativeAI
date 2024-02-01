""" Python 3.11.7 """

""" 1. IMPORTS """
import textwrap
import chromadb
import numpy as np
import pandas as pd
import google.generativeai as genai
import google.ai.generativelanguage as glm
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# CHECK OUT EMBEDDING MODEL LIST
for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    print(m.name) # models/embedding-001

""" 2. PREPARE DATA IN LIST """
loader = PyPDFLoader("Introduction to Machine Learning with Python Chapter 1.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

start = time.time()
pages = loader.load_and_split(text_splitter)
end = time.time()
print("The time of execution of load_and_split is :",(end-start) * 10**3, "ms")
# The time of execution of load_and_split is : 828.1712532043457 ms

chunks_in_list = []

for chunk in pages:
    chunks_in_list.append(chunk.page_content)

len(chunks_in_list) # 65

""" 3. CREATE EMBEDDING USING GOOGLE GEMINI EMBEDDING MODEL """

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    model = 'models/embedding-001'
    title = "Custom query"
    return genai.embed_content(model=model,
                                content=input,
                                task_type="retrieval_document",
                                title=title)["embedding"]

""" 4. SETUP EMBEDDING DB USING CHROMADB  """

def create_chroma_db(documents, name):
  chroma_client = chromadb.Client()
  db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

  for i, d in enumerate(documents):
    db.add(
      documents=d,
      ids=str(i)
    )
  return db

DB_NAME = "chaptr1"

start = time.time()
db = create_chroma_db(chunks_in_list, DB_NAME)
end = time.time()
print("The time of execution of create_chroma_db is :",(end-start) * 10**3, "ms")
# The time of execution of create_chroma_db is : 43316.12300872803 ms

db # Collection(name=chaptr1)

# verify whether embedding creation
pd.DataFrame(db.peek(3))
"""
  ids                                         embeddings metadatas                                          documents  uris  data
0   0  [0.0008179437136277556, -0.06587929278612137, ...      None  CHAPTER 1\nIntroduction\nMachine learning is a...  None  None
1   1  [-0.014457940123975277, -0.10086625069379807, ...      None  Outside of commercial applications, machine le...  None  None
2  10  [0.010867523960769176, -0.08785854279994965, -...      None  Identifying topics in a set of blog posts\nIf ...  None  None  """

""" 5. Getting the relevant document """
def get_relevant_passage(query, db):
#   passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  # TODO: extract only content and eliminate other attributes
  passage = db.query(query_texts=[query], n_results=5, include= ["documents"])
#   return passage
  return str(passage)

query = "why machine learning"

# Perform embedding search
passage = get_relevant_passage(query, db)
type(passage) # <class 'str'>
print(passage)
"""
{'ids': [['2', '0', '1', '10', '6']], 'distances': None, 'metadatas': None, 'embeddings': None, 'documents': [['Why Machine Learning?\nIn the early days of “intelligent” applications, many systems used handcoded rules of\n“if ” and “else” decisions to process data or adjust to user input. Think of a spam filter\nwhose job is to move the appropriate incoming email messages to a spam folder. Y ou\ncould make up a blacklist of words that would result in an email being marked as\n1', 'CHAPTER 1\nIntroduction\nMachine learning is about extracting knowledge from data. It is a research field at the\nintersection of statistics, artificial intelligence, and computer science and is also\nknown as predictive analytics or statistical learning. The application of machine\nlearning methods has in recent years become ubiquitous in everyday life. From auto‐\nmatic recommendations of which movies to watch, to what food to order or which\nproducts to buy, to personalized online radio and recognizing your friends in your\nphotos, many modern websites and devices have machine learning algorithms at their\ncore. When you look at a complex website like Facebook, Amazon, or Netflix, it is\nvery likely that every part of the site contains multiple machine learning models.\nOutside of commercial applications, machine learning has had a tremendous influ‐\nence on the way damendous influ‐\nence on the way data-driven research is done today. The tools introduced in this book\nhave been applied to diverse scientific problems such as understanding stars, finding\ndistant planets, discovering new particles, analyzing DNA sequences, and providing\npersonalized cancer treatments.Y our application doesn’t need to be as large-scale or world-changing as these exam‐\nples in order to benefit from machine learning, though. In this chapter, we will\nexplain why machine learning has become so popular and discuss what kinds of\nproblems can be solved using machine learning. Then, we will show you how to build\nyour first machine learning model, introducing important concepts along the way.\nWhy Machine Learning?\nIn the early days of “intelligent” applications, many systems used handcoded rules of\n“if ” and “else” decisions to process data or adjust to user input. Think of a spam filter', 'Identifying topics in a set of blog posts\nIf you have a large collection of text data, you might want to summarize it and\nfind prevalent themes in it. Y ou might not know beforehand what these topics\nare, or how many topics there might be. Therefore, there are no known outputs.\nWhy Machine Learning? | 3', 'learning algorithms because a “teacher” provides supervision to the algorithms in the\nform of the desired outputs for each example that they learn from. While creating a\ndataset of inputs and outputs is often a laborious manual process, supervised learning\nalgorithms are well understood and their performance is easy to measure. If your\napplication can be formulated as a supervised learning problem, and you are able to\n2 | Chapter 1: Introduction']], 'uris': None, 'data': None} """

""" 6. PREPARE PROMPT """
def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

prompt = make_prompt(query, passage)
print(prompt)
"""
You are a helpful and informative bot that answers questions using text from the reference passage included below.   Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.   However, you are talking to a non-technical audience, so be sure to break down complicated concepts and   strike a friendly and converstional tone.   If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: 'why machine learning'
  PASSAGE: '{ids: [[2, 0, 1, 10, 6]], distances: None, metadatas: None, embeddings: None, documents: [[Why Machine Learning?\nIn the early days of “intelligent” applications, many systems used handcoded rules of\n“if ” and “else” decisions to process data or adjust to user input. Think of a spam filter\nwhose job is to move the appropriate incoming email messages to a spam folder. Y ou\ncould make up a blacklist of words that would result in an email being marked as\n1, CHAPTER 1\nIntroduction\nMachine learning is about extracting knowledge from data. It is a research field at the\nintersection of statistics, artificial intelligence, and computer science and is also\nknown as predictive analytics or statistical learning. The application of machine\nlearning methods has in recent years become ubiquitous in everyday life. From auto‐\nmatic recommendations of which movies to watch, to what food to order or which\nproducts to buy, to personalized online radio and recognizing your friends in your\nphotos, many modern websites and devices have machine learning algorithms at their\ncore. When you look at a complex website like Facebook, Amazon, or Netflix, it is\nvery likely that every part of the site contains multiple machine learning models.\nOutside of commercial applications, machine learning has had a tremendous influ‐\nence on the way data-driven research is done today. The tools introduced in this book, Outside of commercial applications, machine learning has had a tremendous influ‐\nence on the way data-driven research is done today. The tools introduced in this book\nhave been applied to diverse scientific problemcancer treatments.Y our application doesn’t need to be as large-scale or world-changing as these exam‐\nples in order to benefit from machine learning, though. In this chapter, we will\nexplain why machine learning has become so popular and discuss what kinds of\nproblems can be solved using machine learning. Then, we will show you how to build\nyour first machine learning model, introducing important concepts along the way.\nWhy Machine Learning?\nIn the early days of “intelligent” applications, many systems used handcoded rules of\n“if ” and “else” decisions to process data or adjust to user input. Think of a spam filter, Identifying topics in a set of blog posts\nIf you have a large collection of text data, you might want to summarize it and\nfind prevalent themes in it. Y ou might not know beforehand what these topics\nare, or how many topics there might be. Therefore, there are no known outputs.\nWhy Machine Learning? | 3, learning algorithms because a “teacher” provides supervision to the algorithms in the\nform of the desired outputs for each example that they learn from. While creating a\ndataset of inputs and outputs is often a laborious manual process, supervised learning\nalgorithms are well understood and their performance is easy to measure. If your\napplication can be formulated as a supervised learning problem, and you are able to\n2 | Chapter 1: Introduction]], uris: None, data: None}'
    ANSWER: """

""" 7. Q & A WITH CHAPTER 1 """
model = genai.GenerativeModel('gemini-pro')
answer = model.generate_content(prompt)
print(answer.text)
"""
Machine learning is a cutting-edge field that combines statistics, artificial intelligence, and computer science to extract knowledge from data. Its applications are prevalent in everyday life, from personalized recommendations on streaming services, to facial recognition in social media, and even spam filtering in email. Machine learning algorithms are at the core of many modern websites and devices. Its impact extends beyond commercial applications, influencing data-driven research in fields such as astronomy, biology, and medicine. It has revolutionized scientific research, from understanding stars and discovering new planets, to analyzing DNA sequences and providing tailored cancer treatments. Machine learning empowers us to solve complex problems that were previously unsolvable, opening up new possibilities and insights across various domains. """

"""
Tried with many other queries and got "I apologize, but the provided passage does not contain any information about a summary and outlook, thus I cannot respond with an answer."
because embedding search did not provide expected response required for query answer.

Here intention is to use Google embedding model and chromadb, hence not trying to optimize Q&A. """