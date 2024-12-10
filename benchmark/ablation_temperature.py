import json
import os.path

from benchmark.benchmark_class import BenchmarkTest, run_agent_on_benchmark
from benchmark.benchmark_file_util import make_dir_for_benchmark
from person.action.brain_qwen.qwen import QWen
from person.action.brain_chat_glm.chat_glm import ChatGLM2
from person.action.brain_vicuna.vicuna import Vicuna
from person.action.brain_xverse.xverse import XVerse
from person.action.brain.agent import Agent,Agent_self_consistency
from util.given_name import given_name_list
from util.file_util import load_json_file

if __name__=="__main__":
    for model_name in ['gpt-3.5-turbo-1106_tem_0.3','gpt-3.5-turbo-1106_tem_0.4']:
        for person_name in ["homer"]:
            prompt_name = "prompt1"
            profile_version = "profile_v1"
            system_version = "system_v1"
            benchmark_version = "benchmark_v2"
            batch_size = 1
            time_sleep = 0
            if model_name == 'gpt-3.5-turbo-1106':
                batch_size = 1
            elif model_name == 'gpt-4':
                batch_size = 1
            agent = Agent_self_consistency(profile_version=profile_version,
                          system_version=system_version,
                          person_name=person_name,
                          model_name=model_name,
                          )
            run_agent_on_benchmark(
                person_name, prompt_name, profile_version, system_version, benchmark_version, batch_size, agent,
                time_sleep=time_sleep, zero_shot=False)