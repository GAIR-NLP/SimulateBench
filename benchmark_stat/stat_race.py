from benchmark_stat.statistic import Statistic
import os.path

from benchmark_stat.statistic import Statistic
from benchmark_stat.statistic import csv_head_row_models_names_mean, csv_head_row_models_single_name, \
    csv_root_path_ablation
import numpy as np
import csv
from benchmark_stat.statistic import make_csv_file_models_single_person_ablation, \
    make_csv_file_models_all_people_ablation

race_list = ["african", "asian", "Middle_Eastern", "Native_American", "Southern_American", "Northern_European"]
full_name_list = [f"homer_{race}" for race in race_list] + ["homer"]
race_list2 = ["Caucasian", "Pacific Islander", "Indian", "Indigenous Australian"]
if __name__ == "__main__":
    ablation_kind = "race"
    # calculate_fun_mean = "answerable_9"
    benchmark_version = "benchmark_v2"
    profile_version = "profile_v1"
    system_version = "system_v1"
    prompt_name = "prompt1"
    model_list = ["gpt-3.5-turbo-16k", "gpt-4", "chatglm2-6b-32k", "chatglm2-6b", "XVERSE-13B-Chat", "Qwen-7B-Chat",
                  "Qwen-14B-Chat", "vicuna-7b-v1.5-16k", "longchat-7b-16k", "longchat-13b-16k", "longchat-7b-32k-v1.5",
                  "vicuna-13b-v1.5-16k"]
    prompt_kind = ["zero_shot", "few_shot"]
    for prompt in prompt_kind:
        for calculate_fun_name in ["answerable_9", "ration_of_number"]:
            param = {
                "prompt_name": prompt_name,
                "prompt_kind": prompt,
                "model_name_list": model_list,
                "benchmark_version": benchmark_version,
                "profile_version": profile_version,
                "system_version": system_version,
                "full_name_list": full_name_list,
                "ablation_kind": ablation_kind,
                "calculate_mean_fun_name": calculate_fun_name
            }
            make_csv_file_models_all_people_ablation(**param)
            make_csv_file_models_single_person_ablation(**param)
