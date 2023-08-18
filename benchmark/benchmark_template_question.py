from GPTMan.person.action.brain.chat_model import OpenAI


def generated_basic_information_question(question):
    sys_prompt_template = "You are a helpful assistant. You are able to help write questionnaires to ask respondents " \
                          "about some of the things that happen when they're in a certain role." \
                          "You are good at writing one-choice question to fully show all aspects of a person."
    chat_model = OpenAI(system_template=sys_prompt_template)

    # user_text = "I want to write a question about the basic information of a person."
    user_text = f"How may the following question be converted to one-choice questions? add a I don't Know option" \
                f"{question}"

    result = chat_model.run(user_text)
    print(result)
    chat_model.close()


if __name__ == "__main__":
    question = "What is the current status of your relationship with <the individual>? Are you still in contact, have you lost touch, or has the relationship ended in some other way?"
    generated_basic_information_question(question)
