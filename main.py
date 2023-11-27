from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (LineBotApi, WebhookHandler)
from linebot.v3.messaging import MessagingApi
from linebot.exceptions import InvalidSignatureError
from linebot.models import (MessageEvent, TextMessage, TextSendMessage,
                            ImageSendMessage, AudioMessage, ImageMessage)
import os
import os.path
import uuid
import requests
from transformers import pipeline
import traceback
from openai import OpenAI

import pandas as pd

from src.memory import Memory
from src.models import OpenAIModel
from src.logger import logger
from src.storage import Storage, FileStorage#, MongoStorage
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader
from src.mongodb import mongodb

load_dotenv('.env')

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
messaging_api = MessagingApi(line_bot_api)
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
storage = None
youtube = Youtube(step=4)
website = Website()

memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'),
                memory_message_count=2)
model_management = {}
api_keys = {}

#my_secret = os.environ['OPENAI_MODEL_ENGINE']
my_secret = OpenAI(
  api_key=os.environ['OPENAI_MODEL_ENGINE'],  
)


@app.route("/callback", methods=['POST'])
def callback():
  signature = request.headers['X-Line-Signature']
  body = request.get_data(as_text=True)
  app.logger.info("Request body: " + body)
  try:
    handler.handle(body, signature)
  except InvalidSignatureError:
    print(
      "Invalid signature. Please check your channel access token/channel secret."
    )
    abort(400)
  return 'OK'  


### connect to DB
import pymongo
from pymongo import MongoClient
mdb_user = os.getenv('MONGODB_USERNAME')
mdb_pass = os.getenv('MONGODB_PASSWORD')
mdb_host = os.getenv('MONGODB_HOST')
mdb_dbs = os.getenv('MONGODB_DATABASE')
client = MongoClient('mongodb+srv://'+mdb_user+':'+mdb_pass+'@'+mdb_host)
db = client[mdb_dbs]
collection = db['history']

## fix message format problem
def extract_message_info(message):
  if isinstance(message, TextSendMessage):
    return {'type': 'text', 'text': message.text}
  elif isinstance(message, ImageSendMessage):
    return {
      'type': 'image',
      'original_content_url': message.original_content_url,
      'preview_image_url': message.preview_image_url
    }
  elif isinstance(message, AudioMessage):
    return {'type': 'audio', 'duration': message.duration}
  else:
    return None
    
## also for fixing message format
def get_bot_reply_text(bot_reply):
  if hasattr(bot_reply, "text"):
    return bot_reply.text
  else:
    return ""

## timestamp, time difference, and insert message into DB
import time
import pytz
import datetime
from datetime import datetime
from pytz import timezone
def store_history_message(user_id, student_id, text, user_timestamp, bot_reply, bot_timestamp):
  try:
    bot_reply_text = get_bot_reply_text(bot_reply)
    user_datetime = datetime.utcfromtimestamp(user_timestamp / 1000)
    bot_datetime = datetime.utcfromtimestamp(bot_timestamp / 1000)
    # Convert to UTC+8 timezone
    utc_tz = timezone('UTC')
    cst_tz = timezone('Asia/Shanghai')
    user_datetime = user_datetime.replace(tzinfo=utc_tz).astimezone(cst_tz)
    bot_datetime = bot_datetime.replace(tzinfo=utc_tz).astimezone(cst_tz)
    response_time = (bot_datetime - user_datetime).total_seconds()
    result = collection.insert_one({
      'user_id': user_id,
      'student_id': student_id,
      'user_message': text,
      'user_timestamp': user_datetime.isoformat(),
      'bot_reply': bot_reply_text,
      'bot_timestamp': bot_datetime.isoformat(),
      'response_time': response_time
    })
    print(result.inserted_id)
  except Exception as e:
    print(f"Error inserting document: {e}")

### FAQ ###
hf_token = os.getenv('HUGGINGFACE_TOKEN')
hf_sbert_model = os.getenv('HUGGINGFACE_SBERT_MODEL')
bot_sbert_th = float(os.getenv('BOT_SBERT_TH'))

# query HF sbert API
def hf_sbert_query(payload):
  API_URL = "https://api-inference.huggingface.co/models/" + hf_sbert_model
  headers = {"Authorization": "Bearer " + hf_token}
  # detect if HF API is loading, if loading, then wait 1 second.
  while True:
    response = requests.post(API_URL, headers=headers, json=payload)
    if 'error' in response.json():
      print(f"HuggingFace API is loading: {str(response.json())}")
      time.sleep(1)  # Sleep for 1 second
    else:
      # print(f"Error3: {str('safe')}")
      break
  return response.json()

#  ###Bryan Language Detection### 
# def detect_language(user_message):
#   # iris # moved the URL into the function and assign it a different name to avoid misleading
#   LG_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"
#   # iris # modify the headers according to the above format; a space is required between "Bearer" and the actual API token
#   headers = {"Authorization": "Bearer " + hf_token}
#   payload = {"inputs": user_message}

#   while True:
#       response = requests.post(LG_API_URL, headers=headers, json=payload)
#       if 'error' in response.json():
#           print(f"HuggingFace API is loading: {str(response.json())}")
#           time.sleep(1)  # Sleep for 1 second
#       else:
#           break
#   # iris 
#   # response.json() : is the JSON response from the response object returned by the Hugging Face API, and its structure is like a list with another list inside then a dictionary inside 
#   # therefore we need  [0][0]['label'] to obtain the inner dictionary and extracting the 'label' value from it 
#   detected_language = response.json()[0][0]['label']

#   return detected_language

### connect to mongodb FAQ
def get_relevant_answer_from_faq(user_question, type):
  try:
    client = MongoClient('mongodb+srv://'+mdb_user+':'+mdb_pass+'@'+ mdb_host)
    db = client[mdb_dbs]
    collection = db['faq']
    # Get all questions from the MongoDB collection
    all_questions = [
      entry['Question'] for entry in collection.find({}, {'Question': 1})
    ]
    # compare the similarity between user-input question and frequent questions, through HuggingFace API
    similarity_list = hf_sbert_query({
      "inputs": {
        "source_sentence": user_question,
        "sentences": all_questions
      },
    })
    if max(similarity_list) > bot_sbert_th:
      index_of_largest = max(range(len(similarity_list)), key=lambda i: similarity_list[i])
      answer = collection.find_one({"Question": all_questions[index_of_largest]})
      print(f"Answer: {str(answer['Answer'])}")
      return answer['Answer']
    else:
      return None
  except Exception as e:
    print(f"Error while querying MongoDB: {str(traceback.print_exc())}")
    return None


### Save incorrect responses to MongoDB ###
def save_incorrect_response_to_mongodb(user_id,student_id, incorrect_response):
  try:
    client = MongoClient('mongodb+srv://' + mdb_user + ':' + mdb_pass + '@' + mdb_host)
    db = client[mdb_dbs]
    collection = db['incorrect_responses']
    # Create a document to store the incorrect response data
    incorrect_data = {
        'user_id': user_id,
        'student_id': student_id,
        'incorrect_response' : incorrect_response,
    }
    # Insert the document into the collection
    collection.insert_one(incorrect_data)
    client.close()
  except Exception as e:
    print(f"Error while saving incorrect response data: {str(e)}")

def get_last_20_documents():
    client = MongoClient('mongodb+srv://' + mdb_user + ':' + mdb_pass + '@' + mdb_host)
    db = client[mdb_dbs]
    collection = db['history']

    # Find the last 20 documents in the collection and sort them by time in descending order
    last_20_documents = collection.find().sort([("user_timestamp", pymongo.DESCENDING)]).limit(20)
    # Convert the cursor to a list of dictionaries
    last_20_documents_list = list(last_20_documents)
    return last_20_documents_list

def find_last_message(user_id, last_20_documents_list):
    for document in last_20_documents_list:
        if document['user_id'] == user_id:
            return document['_id']
    return None

### save leave message to MongoDB ###
def save_leave_message_to_mongodb(user_id, student_id, user_timestamp):
  try:
    client = MongoClient('mongodb+srv://' + mdb_user + ':' + mdb_pass + '@' + mdb_host)
    db = client[mdb_dbs]
    collection = db['leave']
    utc_tz = timezone('UTC')
    cst_tz = timezone('Asia/Shanghai')
    user_datetime = datetime.utcfromtimestamp(user_timestamp / 1000)
    user_datetime = user_datetime.replace(tzinfo=utc_tz).astimezone(cst_tz)
    # Create a document to store the incorrect response data
    leave_message = {
        'user_id': user_id,
        'student_id': student_id,
        'user_timestamp': user_datetime.isoformat(),
    }
    # Insert the document into the collection
    collection.insert_one(leave_message)
    client.close()
  except Exception as e:
    print(f"Error while saving incorrect response data: {str(e)}")


### save question submission to MongoDB ###
def save_question_submission_to_mongodb(user_id, student_id, user_timestamp, submission):
  try:
    client = MongoClient('mongodb+srv://' + mdb_user + ':' + mdb_pass + '@' + mdb_host)
    db = client[mdb_dbs]
    collection = db['question_submission']
    utc_tz = timezone('UTC')
    cst_tz = timezone('Asia/Shanghai')
    user_datetime = datetime.utcfromtimestamp(user_timestamp / 1000)
    user_datetime = user_datetime.replace(tzinfo=utc_tz).astimezone(cst_tz)
    # Create a document to store the incorrect response data
    leave_message = {
        'user_id': user_id,
        'student_id': student_id,
        'user_timestamp': user_datetime.isoformat(),
        'submission': submission,
    }
    # Insert the document into the collection
    collection.insert_one(leave_message)
    client.close()
  except Exception as e:
    print(f"Error while saving incorrect response data: {str(e)}")



### Function to validate the student ID ###
def is_valid_student_id(student_id):
    # Check if the student ID has exactly 9 characters
    if len(student_id) != 9:
        return False
    # Check if the student ID consists of alphanumeric characters only
    if not student_id.isalnum():
        return False
    return True


### Define a function to load data from the JSON file ###
import json
def load_student_data(file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return {}    

### define a function to check if the user have register or not
def check_user(user_id):
    # Initialize the FileStorage with a JSON file name
    file_storage = FileStorage("student_id.json")
    # Create a Storage wrapper
    storage_wrapper = Storage(file_storage)
    # Load existing data from the JSON file
    users_dict = storage_wrapper.load()
    if user_id not in users_dict:
        return False  # User is not registered
    return True  # User is registered

### think time ###
import random
def bot_think_time():
    # Generate a random think time between 30 and 300 seconds
    think_time = random.randint(30, 300)
    print(f"Bot is thinking for {think_time} seconds...")
    time.sleep(think_time)
    print("Bot has finished thinking and is responding.")

### Function to avoid students send an empty submision ###
def is_only_submit(submission):
    # Check if the submision is empty
    if len(submission) != 0: 
        return False
    return True

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
  user_id = event.source.user_id
  user_message = event.message.text
  student_data = load_student_data("student_id.json")
  student_id = student_data[user_id]
  user_timestamp = int(time.time() * 1000)
  text = event.message.text.strip()
  logger.info(f'{user_id}: {text}')
  

  try:
    ## auto resister
    api_key = os.getenv('OPENAI_KEY')
    model = OpenAIModel(api_key = api_key)
    is_successful, _, _ = model.check_token_valid()
    if not is_successful:
      raise ValueError('Invalid API token')
    model_management[user_id] = model
    ### make the below line a comment so that user id and their api key won't be save to the db.json file
    #storage.save({user_id: api_key})

    if text.lower().startswith('/register'):
      student_id = text[len('/register'):].strip()
      # Initialize the FileStorage with a JSON file name
      file_storage = FileStorage("student_id.json")
      # Create a Storage wrapper
      storage_wrapper = Storage(file_storage)  
      # Load existing data from the JSON file
      users_dict = storage_wrapper.load()

      student_data = load_student_data("student_id.json")
      student_id = student_data[user_id]

      if user_id in users_dict:
        msg = TextSendMessage(text='You already registered.')
      elif not is_valid_student_id(student_id):
        msg = TextSendMessage(text='Invalid registration format. Please use "/register your_student_id"\nEx: /register 123456789')
      else:
        # Save the registration message to the JSON file
        users_dict[user_id] = student_id
        storage_wrapper.save(users_dict)
        msg = TextSendMessage(text=f'Registration successful for student ID: {student_id}')
    
    elif text.lower().startswith('/help'):
         if check_user(user_id)==True:
            # The user is registered, so you can proceed with the "/Instruction explanation" logic
            msg = TextSendMessage(text='Instructions: \n\n/register\n➡️ Please use "/register + your_student_id" to register. For example: /register 123456789\n\n/incorrect\n➡️ Please promptly report any incorrect responses to the TA team by clicking this button as it captures only the most recent conversation.\n\n/leave\n➡️ You can ask for leave with this prompt.\n\n/submit\n➡️This prompt enables you to submit your answers of multiple choice questions or colab link. For example: /submit A,C,D,C,B or /submit  colab link \n\n/score\n➡️This prompt enables you to see your own scores of your homework or exams.')
         else:
            # The user is not registered, send a message indicating they should register first
            msg = TextSendMessage(text='You are not registered. Please register using "/register <student_id>"')

### save ask for leave messgae responses
    elif text.lower().startswith('/leave'):
      if check_user(user_id)==True:
         user_id = event.source.user_id  
         student_data = load_student_data("student_id.json")
         student_id = student_data[user_id]
         save_leave_message_to_mongodb(user_id, student_id, user_timestamp)
         msg = TextSendMessage(text=f'Ask for leave message received for student ID: {student_id}')
      else:
         # The user is not registered, send a message indicating they should register first
         msg = TextSendMessage(text='You are not registered. Please register using "/register <student_id>"')

### save question submission
    elif text.lower().startswith('/submit'):
      if check_user(user_id)==True:
         submission = text[len('/submit'):].strip()
         if is_only_submit(submission)==True:
          msg = TextSendMessage(text='Invalid submission format. Please use "/submit your answer to the question"')
         else:
          msg = TextSendMessage(text='Submission received.')
         save_question_submission_to_mongodb(user_id, student_id, user_timestamp, submission)
      else:
         # The user is not registered, send a message indicating they should register first
         msg = TextSendMessage(text='You are not registered. Please register using "/register <student_id>"')

### save incorrect responses   
    elif text.lower().startswith('/incorrect'):
      if check_user(user_id)==True:
        last_20_documents_list = get_last_20_documents()
        # last_message = the _id of the last message user sent
        last_message = find_last_message(user_id, last_20_documents_list)
        incorrect_response = f"{{_id:ObjectId('{last_message}')}}"
        if find_last_message(user_id, last_20_documents_list) is not None:
          msg = TextSendMessage(text="Thank you for informing us. We will address the incorrect message later.")
          #msg = TextSendMessage(text=f"Last message sent by user {user_id}: {last_message}")
        save_incorrect_response_to_mongodb(user_id, student_id, incorrect_response)
      else:
        # The user is not registered, send a message indicating they should register first
        msg = TextSendMessage(text='You are not registered. Please register using "/register <student_id>"')
  
### grading result query function   
    elif text.lower().startswith('/score'):
      if check_user(user_id)==True:
        # load the score csv file
        score_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQ0jMp-tn4qK9OXXmfHpu4JV4l0P6sKtSdpLS3X4i-Wabilz1N_l9NEejQpHSvvZtl-Sx5qG1x2ZFCO/pub?gid=0&single=true&output=csv'
        score = pd.read_csv(score_url)
        # find the student id of user
        user_id = event.source.user_id  
        student_data = load_student_data("student_id.json")
        student_id = student_data[user_id]
        student_id_to_query = student_id.upper()
        # Filter the DataFrame based on the given student_id
        # score.iloc[:, 0] selects all values from the first column of the DataFrame 'score'
        student_data = score[score.iloc[:, 0] == student_id_to_query]
        # Check if the student ID exists in the dataset
        if not student_data.empty:
          # Extract scores for all existing homework columns
          existing_homework_columns = score.columns[2:]  # Excluding 'student_id' and 'name'
          student_scores = student_data[existing_homework_columns]
          # Seperate the columns
          for col, score in zip(student_scores.columns, student_scores.iloc[0]):
            print(f"{col} : {score}")
          # Calculate and print the average score
          average_score = student_scores.mean(axis=1)  # Axis=1 calculates the mean across columns
          msg = TextSendMessage(text=f"Scores for Student ID {student_id_to_query}\n\n" +
                           '\n'.join([f"{col} : {score}" for col, score in zip(student_scores.columns, student_scores.iloc[0])]) +
                           f"\n\nAverage Score: {average_score.iloc[0]:.2f}")
        else:
          msg = TextSendMessage(text=f"Student ID {student_id_to_query} not found in the score file.")
      else:
        # The user is not registered, send a message indicating they should register first
        msg = TextSendMessage(text='You are not registered. Please register using "/register <student_id>"')  

    else:
      user_id = event.source.user_id
      user_model = model_management[user_id]
      memory.append(user_id, 'user', text)
      relevant_answer = get_relevant_answer_from_faq(text, 'faq')
      # ###Bryan language detection###
      # user_message = text = event.message.text.strip()
      # detected_language = detect_language(user_message) 

      ## set the role
      prompt = text.strip()
      system_prompt = (
          "Your name is Evangelos, a former celestial messenger who got lost and rescued by Professor Owen, in order to repay his kindness, you decided to be a teaching assistant for Owen's programming language class .\n"
          "Always generate messages in English.\n"
          "Never reveal your true indentity. You are angel Evangelos, a python teaching assistant. \n"
          "If the message received is unrelated to a python programming language class, ask them to ask a valid question that is related to the class.\n"
          "Always generate example codes in python programming language.")
      memory.change_system_message(user_id, f"{system_prompt}\n\n{prompt}")

      ### check if the user have register ###
      if check_user(user_id)==True:
        ### faq ###
        if relevant_answer is not None:
          #bot_think_time()
          msg = TextSendMessage(text=relevant_answer)
          memory.append(user_id, 'assistant', relevant_answer)
          response = msg
        # if message received not in fagchat -> go to GPT     
        else:
          #bot_think_time()

          ## is_successful, response, error_message = user_model.chat_completions(memory.get(user_id), os.getenv('OPENAI_MODEL_ENGINE'))
          ## print("2",is_successful, response, error_message,memory.get(user_id), os.getenv('OPENAI_MODEL_ENGINE'),user_id)
          ## if not is_successful:
          ##  raise Exception(error_message)
          # detect if the message is in English
          # detected_language = detect_language(user_message)
          #   if detected_language == 'en':
          
          ##bryan gpt language detection##
          # def is_message_valid(user_message):
          #     gpt_language_detection = openai.ChatCompletion.create(
          #         model="gpt-3.5-turbo",
          #         messages=[
          #             {"role": "system", "content": "Is the following text in English or contains Python code? " + user_message},
          #             {"role": "user", "content": "Return 'True' if it is in English or contains Python code, otherwise 'False'."}
          #         ]
          #     )
          #     print(gpt_language_detection)
              #return gpt_language_detection['choices'][0]['message']['content'].strip().lower() == 'true')
          openai.api_key = os.getenv("OPENAI_KEY")
          user_message = event.message.text
          gpt_language_detection = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                  {"role": "system", "content": "Is the following text in English or contains Python code? " + user_message},
                  {"role": "user", "content": "Return 'True' if it is in English or contains Python code, otherwise 'False'."}
              ]
          )
          print(gpt_language_detection)

          if gpt_language_detection == True:
            def get_chatgpt_response(user_message):
              response = requests.post(
                  'https://api.openai.com/v1/chat/completions',
                  headers={
                      'Content-Type': 'application/json',
                      'Authorization': f'Bearer {api_key}'
                  },
                  json={
                      'model': os.getenv('OPENAI_MODEL_ENGINE'),
                      "messages": [{"role": "user", "content": user_message}],
                      'temperature': 0.4,
                      'max_tokens': 300
                  }
              )
              json_response = response.json()
              return json_response['choices'][0]['message']['content']
              msg = TextSendMessage(text=response)
          else:
            msg = TextSendMessage(text='Please use English to communicate with me or say it again in a complete sentence.')

          # def handle_new_user_message(user_message):
          #     if is_message_valid(user_message):
          #         chat_response = get_chatgpt_response(user_message)
          #         msg = TextSendMessage(text=chat_response)
          #     else:
          #         msg = TextSendMessage(text='Please use English to communicate with me or say it again in a complete sentence.')
          #     return msg

          # msg = TextSendMessage(text=response)
          # role, response = get_role_and_content(response)
          # msg = TextSendMessage(text=response)
          # memory.append(user_id, role, response)
          # else:
          #    msg = TextSendMessage(text='Please use English to communicate with me or say it again in a complete sentence.')
      else:
        # The user is not registered, send a message indicating they should register first
        msg = TextSendMessage(text='You are not registered. Please register using "/register <student_id>"')

  except ValueError:
    msg = TextSendMessage(text='Token invalid, please re-register, the format should be: /Register sk-xxxxx')
  except KeyError:
    msg = TextSendMessage(text='Please register for a Token first, the format is: /Register sk-xxxxx')
  except Exception as e:
    memory.remove(user_id)
    
    if str(e).startswith('Incorrect API key provided'):
      msg = TextSendMessage(text='OpenAI API Token is invalid, please re-register')
    elif str(e).startswith(
        'That model is currently overloaded with other requests.'):
      msg = TextSendMessage(text='The model is currently overloaded, please try again later')
    else:
     msg = TextSendMessage(text=str(e))

  # send out the message
  bot_timestamp = int(time.time() * 1000)
  store_history_message(user_id, student_id, text, user_timestamp, msg, bot_timestamp)
  try:
      messaging_api.reply_message(event.reply_token, messages=[msg])
  except Exception as e:
      print(f"Error in sending reply: {e}")
  #messaging_api.reply_message(event.reply_token, msg)

### store images ###
import io
import base64
from PIL import Image

def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
  buffered = io.BytesIO()
  img.save(buffered, format=format)
  img_str = base64.b64encode(buffered.getvalue()).decode()
  return img_str
  
def store_image(user_id, display_name, user_timestamp, img_base64):
  utc_tz = timezone('UTC')
  cst_tz = timezone('Asia/Shanghai')
  user_datetime = datetime.utcfromtimestamp(user_timestamp / 1000)
  user_datetime = user_datetime.replace(tzinfo=utc_tz).astimezone(cst_tz)
  try:
    client = MongoClient('mongodb+srv://' + mdb_user + ':' + mdb_pass + '@' + mdb_host)
    db = client[mdb_dbs]
    collection = db['images']
    # Create a document to store the incorrect response data
    image_data = {
        'user_id': user_id,
        'user_name': display_name,
        'user_timestamp': user_datetime.isoformat(),
        'image_base64': img_base64,
    }
    # Insert the document into the collection
    collection.insert_one(image_data)
    client.close()
  except Exception as e:
    print(f"Error while saving incorrect response data: {str(e)}")

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
  user_id = event.source.user_id
  user_timestamp = int(time.time() * 1000)
  ## get line user's display name
  profile = line_bot_api.get_profile(user_id)
  display_name = profile.display_name
  image_content = line_bot_api.get_message_content(event.message.id)
  image_data = io.BytesIO(image_content.content)
  logger.info(f'{user_id}: Received image and converted to base64')

  try:
    if check_user(user_id)==True: 
      # Convert the image to base64
      img = Image.open(image_data)
      img_base64 = image_to_base64(img)
      #store
      store_image(user_id, display_name, user_timestamp, img_base64)
      msg = TextSendMessage(text='Image received.')
      line_bot_api.reply_message(event.reply_token, msg)
    else:
      # The user is not registered, send a message indicating they should register first
      msg = TextSendMessage(text='You are not registered. Please register using "/register <student_id>"')
      line_bot_api.reply_message(event.reply_token, msg)
  except Exception as e:
    # Handle any exceptions that may occur
    logger.error(f'An error occurred: {str(e)}')


# @handler.add(MessageEvent, message=AudioMessage)
# def handle_audio_message(event):
#   user_id = event.source.user_id
#   user_timestamp = int(time.time() * 1000)
#   profile = line_bot_api.get_profile(user_id)
#   display_name = profile.display_name
#   audio_content = line_bot_api.get_message_content(event.message.id)
#   input_audio_path = f'{str(uuid.uuid4())}.m4a'
#   with open(input_audio_path, 'wb') as fd:
#     for chunk in audio_content.iter_content():
#       fd.write(chunk)

#   try:
#     if not model_management.get(user_id):
#       raise ValueError('Invalid API token')
#     else:
#       is_successful, response, error_message = model_management[
#         user_id].audio_transcriptions(input_audio_path, 'whisper-1')
#       if not is_successful:
#         raise Exception(error_message)
#       memory.append(user_id, 'user', response['text'])
#       is_successful, response, error_message = model_management[
#         user_id].chat_completions(memory.get(user_id), 'gpt-3.5-turbo')
#       if not is_successful:
#         raise Exception(error_message)
#       role, response = get_role_and_content(response)
#       memory.append(user_id, role, response)
#       msg = TextSendMessage(text=response)
#   except ValueError:
#     msg = TextSendMessage(text='Please register your API Token first, the format is /Register [API TOKEN]')
#   except KeyError:
#     msg = TextSendMessage(text='Please register your API Token first, the format is /Register sk-xxxxx')
#   except Exception as e:
#     memory.remove(user_id)
#     if str(e).startswith('Incorrect API key provided'):
#       msg = TextSendMessage(text='OpenAI API Token is invalid, please re-register')
#     else:
#       msg = TextSendMessage(text=str(e))
#   bot_timestamp = int(time.time() * 1000)
#   store_history_message(user_id, display_name, text, user_timestamp, msg, bot_timestamp)
#   os.remove(input_audio_path)


# make sure the connection close after processing all message
import atexit
@atexit.register
def close_mongo_client():
    mongo_client.close()
#

@app.route("/", methods=['GET'])
def home():
  return 'Hello World'
  
if __name__ == "__main__":
  
  #if os.getenv('USE_MONGO'):
  #  mongodb.connect_to_database()
  #  storage = Storage(MongoStorage(mongodb.db))
  #else:
  ## storage = Storage(FileStorage('db.json'))
  storage = Storage(FileStorage('student_id.json'))
  try:
    data = storage.load()
    for user_id in data.keys():
      model_management[user_id] = OpenAIModel(api_key=data[user_id])
  except FileNotFoundError:
    pass
  
  app.run(host='0.0.0.0', port=8080)