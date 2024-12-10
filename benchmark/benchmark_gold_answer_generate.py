from person.profile.basic_information import load_basic_information
from person.profile.role import load_social_personas
# from person.action.brain.chat_model import OpenAI


def answer(person, question, load_information_func, system_template=None):
    if system_template is None:
        system_template = f"You are a helpful ai assistant, You are helping {person} " \
                          f"to complete a survey given a document written in json format. " \
                          f"The document record all kinds of information about {person}.\n" \
                          f"The question may start with 'you', just answer the question as if you are {person}." \
                          f"You should not answer the question with information not existed in the documents. " \


    information = load_information_func(person_name=person, pure_str=True)
    information_template = f"document is:\n{information}\n, {question}\n"
    llm = OpenAI(system_template=system_template,model_name='gpt-4')

    result = llm.run(user_input=information_template)
    llm.clear()

    return result


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
