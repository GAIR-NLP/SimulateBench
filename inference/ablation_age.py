import os

from inference.run_bench import run_agent_on_benchmark
from inference.model import Llama
age_list = ["1985", "2000", "2010", "2015", "2020"]

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

if __name__ == "__main__":
    # 'Qwen-14B-Chat', "Qwen-7B-Chat" done
    # 'chatglm2-6b-32k', "chatglm2-6b" done
    # "XVERSE-13B-Chat" done
    # "vicuna-7b-v1.5-16k" done
    # "longchat-7b-16k" done
    # longchat-13b-16k done
    # longchat-7b-32k-v1.5 done
    # vicuna-13b-v1.5-16k done
    for model_name in [
        "Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]:
        for person_name in ["homer_" + name for name in age_list]:
            prompt_name = "prompt1"
            profile_version = "profile_v1"
            system_version = "system_v1"
            benchmark_version = "benchmark_v2"
            batch_size = 1
            time_sleep = 0
            if model_name == "gpt-3.5-turbo-16k":
                batch_size = 16
            elif model_name == "gpt-4":
                batch_size = 16
            agent = Llama(
                profile_version=profile_version,
                system_version=system_version,
                person_name=person_name,
                model_name=model_name,
            )
            run_agent_on_benchmark(
                person_name,
                prompt_name,
                profile_version,
                system_version,
                benchmark_version,
                batch_size,
                agent,
                time_sleep=time_sleep,
            )
            # print(agent.run("what is your birthday?"))
