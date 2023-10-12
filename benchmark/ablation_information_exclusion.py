from person.action.brain_xverse.xverse import XVerse
from person.action.brain_chat_glm.chat_glm import ChatGLM2
from person.action.brain_qwen.qwen import QWen
from person.action.brain_vicuna.vicuna import Vicuna
from benchmark.benchmark_class import run_agent_on_benchmark_roles

character_list = ["homer", "rachel", "walter_white", "monica"]

if __name__ == "__main__":
    # 'Qwen-14B-Chat', "Qwen-7B-Chat" done
    # 'chatglm2-6b-32k', "chatglm2-6b" done
    # "XVERSE-13B-Chat" done
    # "vicuna-7b-v1.5-16k" done
    # "longchat-7b-16k" done
    # longchat-13b-16k done
    # longchat-7b-32k-v1.5 done
    # vicuna-13b-v1.5-16k done
    for model_name in ['chatglm2-6b-32k', "chatglm2-6b"]:
        for person_name in character_list:
            prompt_name = "prompt1"
            profile_version = "profile_v1_only_roles"
            system_version = "system_v1"
            benchmark_version = "benchmark_v2"
            batch_size = 1
            time_sleep = 0
            if model_name == 'gpt-3.5-turbo-16k':
                batch_size = 16
            elif model_name == 'gpt-4':
                batch_size = 16
            agent = ChatGLM2(profile_version=profile_version,
                           system_version=system_version,
                           person_name=person_name,
                           model_name=model_name,
                           )
            run_agent_on_benchmark_roles(
                person_name, prompt_name, profile_version, system_version, benchmark_version, batch_size, agent,
                time_sleep=time_sleep)
