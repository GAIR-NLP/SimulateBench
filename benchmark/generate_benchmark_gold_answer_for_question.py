from GPTMan.log.record_cost import record_cost_gpt
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from GPTMan.config.config import settings_system
from GPTMan.person.action.brain.chat_model import MODEL_NAME, generate_open_ai_chat_model
from GPTMan.person.profile.basic_information import load_basic_information
from GPTMan.person.profile.role import load_social_personas


def generate_answer(sys_template, human_text):
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_prompt = HumanMessagePromptTemplate.from_template("{text}")
    human_message = human_prompt.format_messages(text=human_text)

    chat_model = generate_open_ai_chat_model()

    chat_history = sys_prompt.format_messages()
    chat_history.extend(human_message)

    results = chat_model.generate([chat_history])
    result = f"{results.generations[0][0].text}"
    # print(f"{results.generations[0][0].text}\n{'-' * 10}")

    token_usage_prompt = results.llm_output['token_usage']['prompt_tokens']
    token_usage_completion = results.llm_output['token_usage']['completion_tokens']
    record_cost_gpt(MODEL_NAME, input_tokens=token_usage_prompt, output_tokens=token_usage_completion)

    return result


def answer(person, question, load_information_func, system_template=None):
    if system_template is None:
        system_template = f"You are a helpful ai assistant, You are helping {person} " \
                          f"to complete a survey given a document written in json format. " \
                          f"The document record all kinds of information about {person}.\n" \
                          f"The question may start with 'you', just answer the question as if you are {person}." \
                          f"You should not answer the question with information not existed in the documents. " \
                          f"If there is no proper answer based on the provided document, " \
                          f"you must just answer with 'I don't know'."


    information = load_information_func(person_name=person, pure_str=True)
    information_template = f"document is:\n{information}\n, {question}\n"

    return generate_answer(system_template, information_template)


def answer_basic_information(question, person="monica"):
    # question = "What other names have you been called? Who uses each of these names as a nickname?"
    return answer(person=person, question=question, load_information_func=load_basic_information)


def answer_roles_non_relation(question, person="monica"):
    return answer(person=person, question=question, load_information_func=load_social_personas)


def answer_roles_relation(question, person="monica"):
    # question = "How does Rachel Greene interact with you usually?"
    return answer(person=person, question=question, load_information_func=load_social_personas)


if __name__ == "__main__":
    pass
