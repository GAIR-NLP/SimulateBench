

import openai
#openai.api_key = "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"
openai.api_base = "http://openai.plms.ai/v1"

#import openai
# openai.api_key = "sk-dOY5JceVAyP1cjgE17C5E92477994d959d10Cf56A731E7B3"
# openai.api_base = "https://api.aigcbest.top/v1"



openai.api_key = "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"
# openai.api_base = "https://api.chatanywhere.com.cn/v1"


question = """"""


message = [{
              "role": "user",
              "content": question
                              }]

response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=message,
                            max_tokens=50,
                            temperature=0,
                        )
print(response)


