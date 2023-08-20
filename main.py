from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import InvalidSignatureError
from linebot.models import (MessageEvent, TextMessage, TextSendMessage,
                            ImageSendMessage, AudioMessage)
import os
import os.path
import uuid
import requests
import traceback

from src.memory import Memory
from src.models import OpenAIModel
from src.logger import logger
from src.storage import Storage, FileStorage, MongoStorage
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader
from src.mongodb import mongodb

load_dotenv('.env')

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
storage = None
youtube = Youtube(step=4)
website = Website()

memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'),
                memory_message_count=2)
model_management = {}
api_keys = {}

my_secret = os.environ['OPENAI_MODEL_ENGINE']

### connect to DB
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
    
## also for fixing message format problem
def get_bot_reply_text(bot_reply):
  if hasattr(bot_reply, "text"):
    return bot_reply.text
  else:
    return ""

## timestamp, time difference, and insert message into DB
import time
import pytz
from datetime import datetime
from pytz import timezone
def store_history_message(user_id, display_name, text, user_timestamp, bot_reply, bot_timestamp):
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
      'user_message': text,
      'user_name': display_name,
      'user_timestamp': user_datetime.isoformat(),
      'bot_reply': bot_reply_text,
      'bot_timestamp': bot_datetime.isoformat(),
      'response_time': response_time
    })
    print(result.inserted_id)
  except Exception as e:
    print(f"Error inserting document: {e}")

##FAQ
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

  # connect to mongodb FAQ
def get_relevant_answer_from_faq(user_question, type):
  try:
    client = MongoClient('mongodb+srv://'+mdb_user+':'+mdb_pass+'@'+mdb_host)
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


### function to save incorrect responses to MongoDB
def save_incorrect_response_to_mongodb(user_id, user_message, incorrect_response):
  try:
    client = MongoClient('mongodb+srv://' + mdb_user + ':' + mdb_pass + '@' +
                             mdb_host)
    db = client[mdb_dbs]
    collection = db['incorrect_responses']
    # Create a document to store the incorrect response data
    incorrect_data = {
        'user_id': user_id,
        'user_message': user_message,
        'incorrect_response' : incorrect_response,
    }
    # Insert the document into the collection
    collection.insert_one(incorrect_data)
    client.close()
  except Exception as e:
    print(f"Error while saving incorrect response data: {str(e)}")
class Memory:
    def get_latest_user_message(self, user_id):
        # Implement this method to get the latest user message from memory
        pass
    def get_latest_assistant_message(self, user_id):
        # Implement this method to get the latest assistant message from memory
        pass

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
  user_id = event.source.user_id
  user_timestamp = int(time.time() * 1000)
  text = event.message.text.strip()
  logger.info(f'{user_id}: {text}')
  ## get line user's display name
  profile = line_bot_api.get_profile(user_id)
  display_name = profile.display_name
  relevant_answer = get_relevant_answer_from_faq(text, 'faq')

  try:
    ## auto resister
    api_key = os.getenv('OPENAI_KEY')
    model = OpenAIModel(api_key = api_key)
    is_successful, _, _ = model.check_token_valid()
    if not is_successful:
      raise ValueError('Invalid API token')
    model_management[user_id] = model
    storage.save({user_id: api_key})
    
    if text.startswith('/Register'):
       #api_key = text[3:].strip()
       #model = OpenAIModel(api_key=api_key)
       #is_successful, _, _ = model.check_token_valid()
       #if not is_successful:
          #raise ValueError('Invalid API token')
       #model_management[user_id] = model
       #storage.save({user_id: api_key})
       msg = TextSendMessage(text='Token valid, registration successful')

    elif text.startswith('/Instruction explanation'):
      msg = TextSendMessage(
        text=
        'Instructions: \n\n/System Information + Prompt\n👉 Use Prompt to instruct the AI to play a specific role. For example: "Please play the role of someone good at summarizing."\n\n/Clear\n👉 By default, the system keeps a record of the last two interactions. This command clears the history.\n\n/Image + Prompt\n👉 Generate images based on textual prompts with DALL∙E 2 Model.For example: "/Image + cat"\n\n/Voice Input\n👉 Utilizes the Whisper model to convert speech to text and then calls ChatGPT to respond in text.\n\nOther Text Input\n👉 Calls ChatGPT to respond in text for other textual inputs.')
      
    elif text.startswith('/System Information'):
      memory.change_system_message(user_id, text[5:].strip())
      msg = TextSendMessage(text='Input successful')
      
    elif text.startswith('/Clear'):
      memory.remove(user_id)
      msg = TextSendMessage(text='Successfully cleared history messages')

    elif text.startswith('/Image'):
      prompt = text[3:].strip()
      memory.append(user_id, 'user', prompt)
      is_successful, response, error_message = model_management[
        user_id].image_generations(prompt)
      if not is_successful:
        raise Exception(error_message)
      url = response['data'][0]['url']
      msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
      memory.append(user_id, 'assistant', url)
      
      ### save incorrect responses
    elif text.startswith('/Incorrect'):
      # Extract the latest user and assistant messages from the memory
      latest_user_message = memory.get_latest_user_message(user_id)
      latest_assistant_message = memory.get_latest_assistant_message(user_id)
      # Construct the incorrect response data
      user_message = latest_user_message
      incorrect_response = latest_assistant_message
      # Save the incorrect response data to MongoDB
      save_incorrect_response_to_mongodb(user_id, user_message, incorrect_response)
      msg = TextSendMessage(text='Thank you for informing us. The incorrect message has been placed into the database and will be addressed by the development team.')  
      
    # faq 
    elif relevant_answer:
      if relevant_answer == get_relevant_answer_from_faq(text, 'faq'):
         relevant_answer = '(form FAQ Database)\n' + relevant_answer
         msg = TextSendMessage(text = relevant_answer)
         memory.append(user_id, 'assistant', relevant_answer)
         response = relevant_answer
      else:
         relevant_answer = get_relevant_answer_from_faq(text, 'manual')
         if relevant_answer:
          relevant_answer = '(from FAQ Database)\n' + relevant_answer
          msg = TextSendMessage(text=relevant_answer)
          memory.append(user_id, 'assistant', relevant_answer)
          response = relevant_answer
         else:
          msg = TextSendMessage(text='I am sorry, but we are currently unable to find the answer')
          memory.append(user_id, 'assistant', relevant_answer)
          response = relevant_answer
        
    else:
      user_model = model_management[user_id]
      memory.append(user_id, 'user', text)
      url = website.get_url_from_text(text)
      if url:
        if youtube.retrieve_video_id(text):
          is_successful, chunks, error_message = youtube.get_transcript_chunks(
            youtube.retrieve_video_id(text))
          if not is_successful:
            raise Exception(error_message)
          youtube_transcript_reader = YoutubeTranscriptReader(
            user_model, os.getenv('OPENAI_MODEL_ENGINE'))
          is_successful, response, error_message = youtube_transcript_reader.summarize(
            chunks)
          if not is_successful:
            raise Exception(error_message)
          role, response = get_role_and_content(response)
          msg = TextSendMessage(text=response)
        else:
          chunks = website.get_content_from_url(url)
          if len(chunks) == 0:
            raise Exception('Unable to fetch the text from this website')
          website_reader = WebsiteReader(user_model,
          os.getenv('OPENAI_MODEL_ENGINE'))
          is_successful, response, error_message = website_reader.summarize(
            chunks)
          if not is_successful:
            raise Exception(error_message)
          role, response = get_role_and_content(response)
          msg = TextSendMessage(text=response)
      else:
        is_successful, response, error_message = user_model.chat_completions(
          memory.get(user_id), os.getenv('OPENAI_MODEL_ENGINE'))
        if not is_successful:
          raise Exception(error_message)
        role, response = get_role_and_content(response)
        msg = TextSendMessage(text=response)
      memory.append(user_id, role, response)
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
  bot_timestamp = int(time.time() * 1000)
  store_history_message(user_id, display_name, text, user_timestamp, msg, bot_timestamp)
  line_bot_api.reply_message(event.reply_token, msg)
  
@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
  user_id = event.source.user_id
  user_timestamp = int(time.time() * 1000)
  profile = line_bot_api.get_profile(user_id)
  display_name = profile.display_name
  audio_content = line_bot_api.get_message_content(event.message.id)
  input_audio_path = f'{str(uuid.uuid4())}.m4a'
  with open(input_audio_path, 'wb') as fd:
    for chunk in audio_content.iter_content():
      fd.write(chunk)

  try:
    if not model_management.get(user_id):
      raise ValueError('Invalid API token')
    else:
      is_successful, response, error_message = model_management[
        user_id].audio_transcriptions(input_audio_path, 'whisper-1')
      if not is_successful:
        raise Exception(error_message)
      memory.append(user_id, 'user', response['text'])
      is_successful, response, error_message = model_management[
        user_id].chat_completions(memory.get(user_id), 'gpt-3.5-turbo')
      if not is_successful:
        raise Exception(error_message)
      role, response = get_role_and_content(response)
      memory.append(user_id, role, response)
      msg = TextSendMessage(text=response)
  except ValueError:
    msg = TextSendMessage(text='Please register your API Token first, the format is /Register [API TOKEN]')
  except KeyError:
    msg = TextSendMessage(text='Please register your API Token first, the format is /Register sk-xxxxx')
  except Exception as e:
    memory.remove(user_id)
    if str(e).startswith('Incorrect API key provided'):
      msg = TextSendMessage(text='OpenAI API Token is invalid, please re-register')
    else:
      msg = TextSendMessage(text=str(e))
  bot_timestamp = int(time.time() * 1000)
  store_history_message(user_id, display_name, text, user_timestamp, msg, bot_timestamp)
  os.remove(input_audio_path)

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
  if os.getenv('USE_MONGO'):
    mongodb.connect_to_database()
    storage = Storage(MongoStorage(mongodb.db))
  else:
    storage = Storage(FileStorage('db.json'))
  try:
    data = storage.load()
    for user_id in data.keys():
      model_management[user_id] = OpenAIModel(api_key=data[user_id])
  except FileNotFoundError:
    pass
  app.run(host='0.0.0.0', port=8080)
