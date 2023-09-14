import tiktoken
from config.config import settings_system
from person.action.system_setting.system1.chat_template import generate_system_message


def num_tokens_from_chat_messages(messages: list, model="gpt-4-0314") -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_chat_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_chat_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def message_cost(messages: list, model="gpt-3.5-turbo") -> float:
    """Returns the cost of tokens used by messages."""
    models = settings_system.models
    if model not in models:
        raise ValueError(f"Model {model} not found in settings_system.models.")
    tokens = num_tokens_from_chat_messages(messages, model=model)
    cost = tokens * settings_system.cost_per_1k_token[model] / 1000

    print(f'{tokens} prompt tokens counted')
    print(f"{cost} dollars charged at the rate of {settings_system.cost_per_1k_token[model]} per 1,000 tokens")
    return cost


def add(a: int, b: int) -> int:
    return a + b


if __name__ == '__main__':
    content = generate_system_message('monica')[0].content
    messages = [{"message": content, "name": "monica"}]
    print(num_tokens_from_chat_messages(messages=messages))
