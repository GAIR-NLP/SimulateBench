class QuestionGenerator:
    def __init__(self):
        pass

    def generator_basic_information(self):
        sys_template = "You are a helpful assistant. You are able to help write questionnaires to ask respondents " \
                       "about their basic information. " \
                       "You are good at writing one-choice question to fully show all aspects of the person." \
                       "You should provide options for every questions. " \
                       "For every question, add a option: There's not enough information to answer this question"
        user_text = "list 10 one-choice questions (provide options) to ask about the basic information of a person."
        return sys_template, user_text

    def generator_basic_information_v2(self):
        sys_template = "I need your expertise in questionnaire design. I want you to create a set of one-choice " \
                       "questions that will gather basic information about a person. Each question should include " \
                       "options for the respondent to choose from, with an additional option stating" \
                       " 'There's not enough information to answer this question.' " \
                       "Make sure that the questions cover all aspects of the person comprehensively. " \
                       "Remember, the goal is to obtain detailed and accurate responses. " \
                       "Please avoid imposing any assumptions or biases in your questions."
        user_text = "list 15 one-choice questions (provide options) to ask about the basic information of a person."

        return sys_template, user_text
    """def generated_basic_information_question(question):
            sys_prompt_template = "You are a helpful assistant. " \
                                  "You are able to help write questionnaires to ask respondents " \
                                  "about some of the things that happen when they're in a certain role." \
                                  "You are good at writing one-choice question to fully show all aspects of a person."
            chat_model = OpenAI(system_template=sys_prompt_template)

            # user_text = "I want to write a question about the basic information of a person."
            user_text = f"How may the following question be converted to one-choice questions? add a I don't Know option" \
                        f"{question}"

            result = chat_model.run(user_text)
            print(result)
            chat_model.close()"""


if __name__ == "__main__":
    """question = "What is the current status of your relationship with <the individual>? Are you still in contact, have you lost touch, or has the relationship ended in some other way?"
    generated_basic_information_question(question)"""

    """generator = QuestionGenerator()
    generator.generator_basic_information()"""
