from log.record_cost import record_cost_gpt
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from config.config import settings_system
from person.action.brain.chat_model import MODEL_NAME, generate_open_ai_chat_model


def select_dialogue(sys_template, dialogue_list=None):
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_prompt = HumanMessagePromptTemplate.from_template("{text}")
    human_message = human_prompt.format_messages(text=dialogue_list)

    chat_model = generate_open_ai_chat_model()

    chat_history = sys_prompt.format_messages()
    chat_history.extend(human_message)

    results = chat_model.generate([chat_history])
    print(f"{results.generations[0][0].text}\n{'-' * 10}")

    token_usage_prompt = results.llm_output['token_usage']['prompt_tokens']
    token_usage_completion = results.llm_output['token_usage']['completion_tokens']
    record_cost_gpt(MODEL_NAME, input_tokens=token_usage_prompt, output_tokens=token_usage_completion)


def select_dialogue_ethics(person, dialogue_text=None):
    sys_template = "You are a helpful ai assistant. You thoroughly understand the concept of seven deadly sins, " \
                   "including PRIDE, ENVY, WRATH, SLOTH, GREED, GLUTTONY, and LUST.\n" \
                   "If there's a dialogue here, it's between two people. " \
                   "You are able to identify if one of them has done one of the seven deadly sins by which one.\n" \
                   "Present your result with the following format:\n" \
                   "result:<Yes or No>, person: <which person>, " \
                   "sin:<if you say Yes, list which one of the seven deadly sins>, " \
                   "reason:<how do you get your result>.\n"

    dialogue = "Rachel: Oh, Monica. You are not still going over that thing.\n" \
               "Monica: This woman's living my life.\n" \
               "Rachel: What?\n" \
               "Monica: She's living my life, and she's doing it better than me! Look at this, look. She buys " \
               "tickets for plays that I wanna see. She, she buys clothes from stores that I'm intimidated by the " \
               "sales people. She spent three hundred dollars on art supplies.\n" \
               "Rachel: You're not an artist.\n" \
               "Monica: Yeah, well I might be if I had the supplies! I mean, I could do all this stuff. Only I don't.\n" \
               "Rachel: Oh, Monica, c'mon, you do cool things.\n" \
               "Monica: Oh really? Okay, let's compare, shall we.\n" \
               "Rachel: Oh, it's so late for 'Shall we'...\n" \
               "Monica: Do I go horseback riding in the park? Do I take classes at the New School?\n" \
               "Rachel: Nooo...\n" \
               "Monica: This is so unfair! She's got everything I want, and she doesn't have my mother.\n"

    dialogue_format = f"{dialogue_text}\n Your result for {person} is:"

    select_dialogue(sys_template, dialogue_list=dialogue_format)


def select_dialogue_emotions(person, dialogue_text=None):
    sys_template = "You are a helpful ai assistant. You thoroughly understand the concept of 6 Types of Basic Emotions, " \
                   "including Happiness, Sadness, Fear, Disgust, Anger, Surprise.\n" \
                   "If there's a dialogue here, it's between two people. " \
                   "You are able to identify if one of them has the emotion of the six basic emotions by which one.\n" \
                   "Present your result with the following format:\n" \
                   "result:<Yes or No>, person: <which person>, " \
                   "emotion:<if you say Yes, list which one of the six emotion>, reason:<how do you get your result>.\n"

    dialogue = "environments:[Scene: Chandler's bedroom, he is giving Monica a massage.]\n" \
               "Monica: I can't believe we've never done this before! It's sooo good! So good for Monica!\n" \
               "environments:(Chandler picks up the timer being used and turns it to zero at which it chimes.)\n" \
               "Chandler: Oh! Look at that, time's up! My turn!\n" \
               "Monica: That was a half an hour?\n" \
               "Chandler: It's your timer.\n" \
               "environments:(They change places.)\n" \
               "Monica: Y'know, I don't like to brag about it, but I give the best massages!\n" \
               "Chandler: All right, then massage me up right nice!\n" \
               "environments:(She starts the massage, only she is doing extremely hard and Chandler is gasping in pain.)\n" \
               "Chandler: Ah! Ahh!! Ahh!!\n" \
               "Monica: It's so good, isn't it?\n" \
               "Chandler: It's so good I don't know what I've done to deserve it!\n" \
               "Monica: Say good-bye to sore muscles!\n" \
               "Chandler: Good-bye muscles!!\n"

    dialogue_format = f"{dialogue}\n Your result for {person} is:"

    select_dialogue(sys_template, dialogue_list=dialogue_format)


def select_dialogue_personality(person, dialogue_text=None):
    sys_template = "You are a helpful ai assistant. You thoroughly understand the concept of " \
                   "Big Five personality traits, including " \
                   "openness to experience (inventive/curious vs. consistent/cautious)," \
                   "conscientiousness (efficient/organized vs. extravagant/careless)," \
                   "extraversion (outgoing/energetic vs. solitary/reserved)," \
                   "agreeableness (friendly/compassionate vs. critical/rational)," \
                   "neuroticism (sensitive/nervous vs. resilient/confident).\n" \
                   "If there's a dialogue here, it's between two people. " \
                   "You are able to identify if one of them show the personality of the Big Five personality traits " \
                   "by which one.\n" \
                   "Present your result with the following format:\n" \
                   "result:<Yes or No>, person: <which person>, " \
                   "personality:<if you say Yes, list which one of the Big Five personality traits and its category>, " \
                   "reason:<how do you get your result>.\n"

    dialogue = "environments:[Scene: Chandler's bedroom, he is giving Monica a massage.]\n" \
               "Monica: I can't believe we've never done this before! It's sooo good! So good for Monica!\n" \
               "environments:(Chandler picks up the timer being used and turns it to zero at which it chimes.)\n" \
               "Chandler: Oh! Look at that, time's up! My turn!\n" \
               "Monica: That was a half an hour?\n" \
               "Chandler: It's your timer.\n" \
               "environments:(They change places.)\n" \
               "Monica: Y'know, I don't like to brag about it, but I give the best massages!\n" \
               "Chandler: All right, then massage me up right nice!\n" \
               "environments:(She starts the massage, only she is doing extremely hard and Chandler is gasping in pain.)\n" \
               "Chandler: Ah! Ahh!! Ahh!!\n" \
               "Monica: It's so good, isn't it?\n" \
               "Chandler: It's so good I don't know what I've done to deserve it!\n" \
               "Monica: Say good-bye to sore muscles!\n" \
               "Chandler: Good-bye muscles!!\n"

    dialogue_format = f"{dialogue}\n Your result for {person} is:"

    select_dialogue(sys_template, dialogue_list=dialogue_format)


if __name__ == '__main__':
    select_dialogue_personality(person="Monica")
