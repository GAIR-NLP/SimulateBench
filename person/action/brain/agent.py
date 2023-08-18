from GPTMan.person.action.brain.template.chat_template import generate_system_message, generate_human_message
from GPTMan.person.action.brain.chat_model import generate_open_ai_chat_model
from GPTMan.person.profile.profile import load_name
from GPTMan.person.action.brain.chat_model import MODEL_NAME
from GPTMan.log.record_cost import record_cost_gpt

PERSON_NAME = "monica"
CHAT_HISTORY = generate_system_message(person_name=PERSON_NAME)  # [system,example,human,ai,system,example]
NAME = load_name(person_name=PERSON_NAME)


class Agent:
    def __init__(self, person_name,model_name=MODEL_NAME):
        self.chat_model = generate_open_ai_chat_model(model_name=model_name)
        self.chat_history = generate_system_message(person_name=person_name)  # [system,example,human,ai,system,example]

        self.person_name = person_name
        self.name = load_name(person_name=person_name)

        self.total_tokens_prompt = 0
        self.total_tokens_output = 0

        self.model_name = model_name

    def run(self, user_input):
        user_message = generate_human_message(user_input)
        self.chat_history.extend(user_message)
        results = self.chat_model.generate([self.chat_history])
        token_usage_prompt = results.llm_output['token_usage']['prompt_tokens']
        token_usage_completion = results.llm_output['token_usage']['completion_tokens']
        self.total_tokens_output += token_usage_completion
        self.total_tokens_prompt += token_usage_prompt
        self.chat_history.append(results.generations[0][0].message)

        return results.generations[0][0].text

    def clear_message(self):
        self.chat_history = generate_system_message(person_name=self.person_name)
        record_cost_gpt(self.model_name, self.total_tokens_prompt, self.total_tokens_output)
        self.total_tokens_prompt = 0
        self.total_tokens_output = 0




if __name__ == "__main__":
    agent = Agent(person_name=PERSON_NAME)
    print(f"hello, I am {NAME}, what can I do for you? If you want to quit, please input 'quit' or 'q'\n{'-' * 10}")
    user_input = input("user:")
    try:
        while user_input != 'quit' and user_input != 'q':
            print(agent.run(user_input))
            user_input = input("user:\n")

    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        print("bye")
        agent.clear_message()
