from benchmark_stat.statistic import make_csv_file_models_single_person_ablation, make_csv_file_models_all_people_ablation

age_list = ["1985", "2000", "2010", "2015", "2020"]
full_name_list = [f"homer_{year}" for year in age_list] + ["homer"]
age_list2 = ["1975", "1980", "1990", "1995", "2005"]
if __name__ == "__main__":
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
                "ablation_kind": "age",
                "calculate_mean_fun_name": calculate_fun_name
            }
            make_csv_file_models_all_people_ablation(**param)
            make_csv_file_models_single_person_ablation(**param)

