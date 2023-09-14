from person.action.brain.chat_model import OpenAI
from util.file_util import load_json_file
from person.profile.basic_information import load_basic_information
from person.profile.role import load_social_personas
import json

ROOT_PATH = "/home/yxiao2/pycharm/GPTMan/db/profile/template/{profile_type}.json"
ROOT_PATH_CHARACTER = "/home/yxiao2/pycharm/GPTMan/db/profile"


class ProfileGenerator:
    def __init__(self):
        self.template = None
        self.chat_model = None

        pass

    def init_chatbot(self, profile_type):
        template_path = ROOT_PATH.format(profile_type=profile_type)
        template_file = load_json_file(template_path)
        template_str = json.dumps(template_file)

        if profile_type == "basic_information":
            example = load_basic_information(pure_str=True)
        elif profile_type == "roles":
            example = load_social_personas(pure_str=True)
        else:
            raise Exception("Profile type not supported")

        template = "You are a helpful assistant, you can extract information of film and television characters, and output the information according to a fixed format.\n" \
                   "the following is the example:\n" \
                   "{example}" \
                   "the following is the template:\n" \
                   "{template}".format(template=template_str, example=example)

        self.template = template

        self.chat_model = OpenAI(system_template=self.template, model_name="gpt-3.5-turbo-16k")

    def generate_profile(self, user_information, profile_type, user_name):
        self.init_chatbot(profile_type=profile_type)
        input_ = f"extract the information of {user_information}."
        results = self.chat_model.run_agent_on_benchmark(input_)
        self.chat_model.clear()

        write_path = F"{ROOT_PATH_CHARACTER}/{user_name}/profile_v1/{profile_type}.json"
        with open(write_path, "w") as f:
            json.dump(results, f, indent=4)


class PersonaGenerator:
    def __init__(self):
        self.template = None
        self.chat_model = None

        pass

    def init_chatbot(self, profile_type, role):
        template_path = ROOT_PATH.format(profile_type=profile_type)
        template_file = load_json_file(template_path)["roles"][role]
        template_str = json.dumps(template_file)

        if profile_type == "basic_information":
            example = load_basic_information(pure_str=True)
        elif profile_type == "roles":
            example = load_social_personas(pure_str=True)
        else:
            raise Exception("Profile type not supported")

        example = json.loads(example)
        example = example["roles"][role]

        template = "You are a helpful assistant, you can extract information of film and television characters, and output the information according to a fixed format.\n" \
                   "the following is the example:\n" \
                   "{example}" \
                   "the following is the template:\n" \
                   "{template}".format(template=template_str, example=example)

        self.template = template

        self.chat_model = OpenAI(system_template=self.template, model_name="gpt-3.5-turbo-16k")

    def generate_persona(self, user_information, profile_type, role):
        self.init_chatbot(profile_type=profile_type, role=role)
        input_ = f"extract the information of {user_information}."
        results = self.chat_model.run_agent_on_benchmark(input_)
        self.chat_model.clear()

        print(results)


class PersonaGeneratorWithPlainQuestion:
    def __init__(self):
        self.template = None
        self.chat_model = None

        pass

    def init_chatbot(self):
        template = "You are a helpful assistant, you can extract information of characters in TV show.\n"

        self.template = template

        self.chat_model = OpenAI(system_template=self.template)

    def generate_persona(self, user_information, information_part, role, print_=True):
        self.init_chatbot()
        if information_part == "routines_or_habits":
            input_ = f"what is the daily routines/habits of {user_information} as a {role}"
        elif information_part == "general experiences":
            input_ = f"what is the important experiences of {user_information} as a {role}."
        elif information_part == "goals_or_plans":
            input_ = f"what is the goals/plans of {user_information} as a {role}"
        else:
            input_ = f"what is the {information_part} of {user_information} as a {role}."

        if not print_:
            print(self.template + input_)
            return

        results = self.chat_model.run_agent_on_benchmark(input_)
        self.chat_model.clear()
        print(results)

    def ask_relationship(self, target_person, user, topic):
        self.init_chatbot()
        if topic == "experience":
            input_ = f"what is {user}'s the important experiences with {target_person}"
        else:
            input_ = f"what is {user}'s {topic} towards {target_person}"

        print(self.template + input_)


def generate_question_to_extract_information_from_openai(user_information, role, user_name, target_person):
    """
    generate question to extract role/persona information from chatgpt
    Args:
        user_information: character or movies
        role: what is the social role of the character
        user_name:character name
        target_person:the person that the character has a relationship with

    Returns:

    """
    persona_generator = PersonaGeneratorWithPlainQuestion()

    for topic in ["routines_or_habits", "general experiences", "goals_or_plans", "characteristics"]:
        persona_generator.generate_persona(user_information, topic, role, print_=False)
    for topic in ["familiarity", "judgement", "affection", "relationship", "behavior_pattern"]:
        persona_generator.ask_relationship(target_person=target_person, user=user_name, topic=topic)


if __name__ == "__main__":
    # generate basic information of the character
    #profile_generator = ProfileGenerator()
    #profile_generator.generate_profile("Homer Simpson in The Simpsons", "basic_information", "homer")

    # generate question to extract information from chatgpt
    user_information = "Homer Simpson in The Simpsons"
    role = "family_member"
    user_name = "homer"
    target_person = "Marge Simpson"
    generate_question_to_extract_information_from_openai(user_information, role, user_name, target_person)
