# https://ai.google.dev/api/python/google/generativeai

# pip install google-generativeai
# pip install python-dotenv

import google.generativeai as genai
import os
from dotenv import load_dotenv

# LOAD API KEY FROM ENVIRONMENT VARIABLE
load_dotenv()
google_gemini_secret_key = os.getenv("GOOGLE_API_KEY")

# ***** ACCESS GEMINI PRO LLM MODEL *****
model = genai.GenerativeModel('gemini-pro')
    #
response = model.generate_content('what can you do for me')

print(response.text)

# In case API failed to return a result, it was blocked due to saftey concerns regarding the prompt, let's check
response.prompt_feedback

"""
safety_ratings {
  category: HARM_CATEGORY_SEXUALLY_EXPLICIT
  probability: NEGLIGIBLE
}
safety_ratings {
  category: HARM_CATEGORY_HATE_SPEECH
  probability: NEGLIGIBLE
}
safety_ratings {
  category: HARM_CATEGORY_HARASSMENT
  probability: NEGLIGIBLE
}
safety_ratings {
  category: HARM_CATEGORY_DANGEROUS_CONTENT
  probability: NEGLIGIBLE
}
"""

# MODEL GENERATES MULTIPLE RESPONSES FOR A PROMPT, CHOOSE WHATEVER IS APPROPRIATE
response.candidates
"""
[index: 0
content {
  parts {
    text: "As a conversational AI or chatbot, I have a wide range of capabilities to assist you with various tasks and provide information. Here are some things I can do for you:\n\n**1. Answer Questions:**\n   - I can answer your questions on a variety of topics, including general knowledge, science, history, culture, and current events. Just ask me what you\'re curious about.\n\n**2. Provide Information:**\n   - I can provide you with information on a wide range of topics, such as news, weather, sports, travel destinations, recipes, and more. Just ask me what you\'d like to know.\n\n**3. Language Translation:**\n   - I can translate text and speech between over 100 languages, making it easier for you to communicate with people from different linguistic backgrounds.\n\n**4. Text Summarization:**\n   - If you have a long document or article, I can summarize it for you, providing you with the key points and main ideas in a concise format.\n\n**5. Generate Creative Content:**\n   - I can help you generate creative content, such as stories, poems, song lyrics, and even code snippets. Just provide me with some ideas or prompts.\n\n**6. Provide Definitions and Explanations:**\n   - If you come across a word or concept that you\'re unsure about, I can provide you with definitions, explanations, and examples to help you understand it better.\n\n**7. Math and Calculations:**\n   - I can perform basic and advanced mathematical operations, including arithmetic, algebra, geometry, and statistics. Just ask me to solve a math problem or perform a calculation.\n\n**8. Unit Conversions:**\n   - I can convert between different units of measurement, such as temperature, length, weight, volume, and more. Just specify the values and units you want to convert.\n\n**9. Date and Time Calculations:**\n   - I can help you with date and time calculations, such as adding or subtracting days, weeks, or months, or converting between time zones.\n\n**10. Entertainment and Games:**\n    - I can provide you with entertainment and games, such as trivia quizzes, word games, riddles, and puzzles. Just ask me to start a game or quiz.\n\n**11. Set Reminders and Alarms:**\n    - I can set reminders and alarms for you, ensuring that you don\'t forget important tasks or appointments.\n\n**12. Recommendations:**\n    - If you\'re looking for recommendations on movies, books, music, or travel destinations, I can provide you with personalized suggestions based on your preferences.\n\n**13. Fact-Checking and Information Verification:**\n    - I can help you verify the accuracy of information and facts by cross-referencing multiple sources and providing reliable information.\n\n**14. News and Updates:**\n    - I can keep you updated on the latest news and current events, providing you with headlines and summaries from reputable sources.\n\n**15. Fun Facts and Trivia:**\n    - I can share fun facts and trivia on a wide range of topics, providing you with interesting and entertaining information.\n\nRemember, I am continuously learning and expanding my capabilities. If you have a specific request or question, feel free to ask me. I\'ll do my best to provide you with accurate and helpful information."
  }
  role: "model"
}
finish_reason: STOP
safety_ratings {
  category: HARM_CATEGORY_SEXUALLY_EXPLICIT
  probability: NEGLIGIBLE
}
safety_ratings {
  category: HARM_CATEGORY_HATE_SPEECH
  probability: NEGLIGIBLE
}
safety_ratings {
  category: HARM_CATEGORY_HARASSMENT
  probability: NEGLIGIBLE
}
safety_ratings {
  category: HARM_CATEGORY_DANGEROUS_CONTENT
  probability: NEGLIGIBLE
}
] """

# STREAM RESPONSE
response = model.generate_content('what can you do for me', stream=True)

for chunk in response:
  print(chunk.text)
  print("_"*80)

"""
As a large language model, I have a wide range of capabilities and can assist
________________________________________________________________________________
 you with a variety of tasks. Here are some things I can do for you:

1. **Answer Your Questions:** I can answer your questions on a
________________________________________________________________________________
 wide range of topics, including general knowledge, science, history, current events, and more.

2. **Provide Information:** I can provide you with detailed information on specific topics, such as historical events, scientific concepts, cultural practices, and more.

3. **Generate Text:** I can generate different types of
________________________________________________________________________________
 text, including creative stories, essays, articles, poems, scripts, and more.

4. **Translate Languages:** I can translate text between over 100 languages, including English, Spanish, French, Chinese, and more.

5. **Summarize and Paraphrase:** I can summarize long texts or articles into shorter, more concise versions, or I can paraphrase text to convey the same information in different words.

6. **Fact-Checking and Research:** I can help you verify facts, gather research, and provide reliable sources of information.

7. **Language Generation:** I can generate natural language in response
________________________________________________________________________________
 to a variety of prompts, including text-based commands, questions, and conversations.

8. **Coding and Programming:** I can write and debug code in various programming languages, including Python, Java, C++, and more.

9. **Data Analysis and Visualization:** I can analyze large datasets, extract insights, and generate visualizations such as charts, graphs, and maps.

10. **Creative Writing:** I can help you write creative content, such as short stories, poems, song lyrics, and other forms of creative expression.

11. **Personal Assistant Tasks:** I can help you with various personal assistant tasks, such as setting alarms, reminders, managing schedules, and providing weather updates.

12. **Ideation and Problem-Solving:** I can help you brainstorm ideas, solve problems, and generate solutions to complex challenges.

13. **Entertainment and Games:** I can engage in entertaining conversations, play games, and provide trivia and quizzes for your enjoyment.

14. **Language Learning:** I can help you learn new languages by providing translations, grammar explanations, and interactive language exercises.

15. **Technical Support:** I can provide technical support for various software, devices, and online services.

These are just a few
________________________________________________________________________________
 examples of the things I can do for you. Feel free to ask me questions or give me instructions, and I will do my best to assist you.
________________________________________________________________________________ """

# LIST MODELS
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

"""
models/gemini-pro
models/gemini-pro-vision """

# gemini-pro: optimized for text-only prompts.
# gemini-pro-vision: optimized for text-and-images prompts.

# ***** ACCESS GEMINI PRO VISION LLM MODEL *****

import PIL.Image

img = PIL.Image.open('image.jpg')
img
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1500x933 at 0x1DE802B7A90>

model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(img)

response.text
"""
' A set of thin line icons with captions representing perception,
cognition, understanding, vision, reality, knowledge, memory, and sensory system.'
 """
