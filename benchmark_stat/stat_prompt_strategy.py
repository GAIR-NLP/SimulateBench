from benchmark_stat.statistic import make_csv_file_models_single_person
from benchmark.ablation_character import character_list

full_name_list = ["homer"]
if __name__ == "__main__":
    for character_name in full_name_list:
        benchmark_version = "benchmark_v2"
        profile_version = "profile_v1"
        system_version = "system_v1"

        model_list = ["gpt-3.5-turbo-16k", "gpt-4", "chatglm2-6b-32k", "chatglm2-6b", "XVERSE-13B-Chat", "Qwen-7B-Chat",
                      "Qwen-14B-Chat", "longchat-7b-16k", "longchat-13b-16k", "longchat-7b-32k-v1.5",
                      "vicuna-7b-v1.5-16k",
                      "vicuna-13b-v1.5-16k"
                      ]
        prompt_kind = [("zero_shot", "prompt2"),("zero_shot", "prompt1"),
                       ("few_shot", "prompt2"), ("few_shot", "prompt3"),("few_shot", "prompt1")
                       ]
        for prompt, prompt_name in prompt_kind:
            param = {
                "prompt_name": prompt_name,
                "prompt_kind": prompt,
                "model_name_list": model_list,
                "benchmark_version": benchmark_version,
                "profile_version": profile_version,
                "system_version": system_version,
                "character_name": character_name,
                "calculate_mean_fun_name": "ration_of_number"
            }
            make_csv_file_models_single_person(**param)
