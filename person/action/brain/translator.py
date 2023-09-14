from person.action.brain.chat_model import OpenAI


class BetterPrompt:
    def __init__(self):
        self.sys_prompt = "Refine the following prompt requirement " \
                          "to help LLM better understand the task and more effectively."
        self.agent = OpenAI(system_template=self.sys_prompt)

    def run(self, text):
        result = self.agent.run(text)
        self.agent.clear()
        return result


class Rewrite:
    def __init__(self, number=3, model_name="gpt-4"):
        self.model_name = model_name
        self.agent = None
        self.sys_prompt = None
        self.rewrite_times = number
        self.load_agent()

    def load_agent(self):
        self.sys_prompt = f"Please rewrite the given sentence in {self.rewrite_times} " \
                          f"different ways, using varied vocabulary and " \
                          f"sentence structures while maintaining the original meaning. " \
                          f"list the answer in the format of json list as " \
                          f"[<sentence1>..]"
        self.agent = OpenAI(system_template=self.sys_prompt, model_name=self.model_name)

    def run(self, text):
        result = self.agent.run(text)
        self.agent.clear()
        return result

    def set_rewrite_times(self, number):
        self.rewrite_times = number
        self.load_agent()

    def clear(self):
        pass


if __name__ == "__main__":
    # better_prompt = BetterPrompt()
    # print(better_prompt.run("I want you to rewrite sentence 10 times with different words."))
    # print(better_prompt.run("Refine the following prompt requirement to help LLM better understand the task and more effectively."))

    rewrite = Rewrite()
    print(rewrite.run("Monica is the middle child of Jack and Judy Johnson, and the younger sister of Ross Johnson."))
