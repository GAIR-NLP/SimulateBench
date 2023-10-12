from person.action.brain_xverse.xverse import XVerse
from person.action.brain_chat_glm.chat_glm import ChatGLM2
from person.action.brain_qwen.qwen import QWen
from person.action.brain_vicuna.vicuna import Vicuna
from benchmark.benchmark_class import run_agent_on_benchmark_single

prompt_strategy_list = [
    {
        "prompt_name": "prompt2",
        "prompt_kind": "few_shot"

    },
    {
        "prompt_name": "prompt3",
        "prompt_kind": "few_shot"

    },
    {
        "prompt_name": "prompt2",
        "prompt_kind": "zero_shot"
    }
]

if __name__ == "__main__":
    # 'Qwen-14B-Chat', "Qwen-7B-Chat" done
    # 'chatglm2-6b-32k', "chatglm2-6b" done
    # "XVERSE-13B-Chat" done
    # "vicuna-7b-v1.5-16k" done
    # "longchat-7b-16k" done
    # longchat-13b-16k done
    # longchat-7b-32k-v1.5 done
    # vicuna-13b-v1.5-16k done
    for model_name in ["XVERSE-13B-Chat"]:
        for person_name in ["homer"]:
            for prompt_strategy in prompt_strategy_list:
                profile_version = "profile_v1"
                system_version = "system_v1"
                benchmark_version = "benchmark_v2"
                batch_size = 1
                time_sleep = 0
                if model_name == 'gpt-3.5-turbo-16k':
                    batch_size = 16
                elif model_name == 'gpt-4':
                    batch_size = 16
                agent = XVerse(profile_version=profile_version,
                               system_version=system_version,
                               person_name=person_name,
                               model_name=model_name,
                               )
                run_agent_on_benchmark_single(
                    person_name=person_name, prompt_name=prompt_strategy["prompt_name"],
                    profile_version=profile_version, system_version=system_version, benchmark_version=benchmark_version,
                    batch_size=batch_size, agent=agent, prompt_kind=prompt_strategy["prompt_kind"],
                    time_sleep=time_sleep)
