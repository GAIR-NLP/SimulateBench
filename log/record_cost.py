from GPTMan.config.config import settings_system
from GPTMan.log.logger import logger


def record_cost_gpt(model_name, input_tokens, output_tokens):
    input_price = settings_system['cost_per_1k_token'][model_name]['input']
    output_price = settings_system['cost_per_1k_token'][model_name]['output']

    input_cost = (input_tokens * input_price) / 1000
    output_cost = (output_tokens * output_price) / 1000
    total_cost = (input_cost + output_cost)
    logger.info(
        f"cost: {round(total_cost, 4)}. "
        f"prompt tokens:{input_tokens}({input_price}/1k tokens). "
        f"completion tokens:{output_tokens}({output_price}/1k tokens)."
    )


if __name__=="__main__":
    record_cost_gpt('gpt-4', 1000, 1000)