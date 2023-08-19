import openai
from GPTMan.config.config import settings_system
openai.api_key = settings_system['OPENAI_API_KEY']  # supply your API key however you choose
openai.api_base = "http://openai.plms.ai/v1"
print('startssss')
completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": "Hello world"}])
print(completion.choices[0].message.content)