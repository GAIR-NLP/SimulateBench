from benchmark_stat.statistic import make_csv_file_models_single_person, make_csv_file_models_all_people_ablation
from util.education import education_list

full_name_list = [f"homer_education_{education}" for education in education_list] + ["homer"]

if __name__ == "__main__":
    benchmark_version = "benchmark_v2"
    profile_version = "profile_v1"
    system_version = "system_v1"

    prompt_name = "prompt1"

    model_list = ["gpt-3.5-turbo-16k", "gpt-4", "chatglm2-6b-32k", "chatglm2-6b", "XVERSE-13B-Chat", "Qwen-7B-Chat",
                  "Qwen-14B-Chat"]
    prompt_kind = ["zero_shot", "few_shot"]
    for prompt in prompt_kind:
        param = {
            "prompt_name": prompt_name,
            "prompt_kind": prompt,
            "model_name_list": model_list,
            "benchmark_version": benchmark_version,
            "profile_version": profile_version,
            "system_version": system_version,
            "full_name_list": full_name_list,
            "ablation_kind": "age",
            "calculate_mean_fun_name": "answerable_9"
        }
        make_csv_file_models_all_people_ablation(**param)
        make_csv_file_models_single_person(**param)
