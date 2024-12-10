from util.given_name import given_name_list
from benchmark_stat.statistic import Statistic
from tqdm import tqdm
def accuracy_single(reslults_dict):
    ba_an = []
    ba_un = []
    no_an = []
    no_un = []
    ro_an = []
    ro_un = []
    
    for model_name in reslults_dict.keys():
        ba_an.append(reslults_dict[model_name]["ba_an"])
        ba_un.append(reslults_dict[model_name]["ba_un"])
        no_an.append(reslults_dict[model_name]["no_an"])
        no_un.append(reslults_dict[model_name]["no_un"])
        ro_an.append(reslults_dict[model_name]["ro_an"])
        ro_un.append(reslults_dict[model_name]["ro_un"])
    
    results=[]
    for list_ in [ba_an,ba_un,no_an,no_un,ro_an,ro_un]:
        results_=[]
        for value0, in zip(list_[0]):
            results_.append((0+value0))
        results.append(results_)
    
    accuracy=[]
    for list_ in results:
        accuracy.append(sum(list_)/len(list_))
    
    print(accuracy)
def accuracy_self_consistency(reslults_dict):
    ba_an = []
    ba_un = []
    no_an = []
    no_un = []
    ro_an = []
    ro_un = []
    
    for model_name in reslults_dict.keys():
        ba_an.append(reslults_dict[model_name]["ba_an"])
        ba_un.append(reslults_dict[model_name]["ba_un"])
        no_an.append(reslults_dict[model_name]["no_an"])
        no_un.append(reslults_dict[model_name]["no_un"])
        ro_an.append(reslults_dict[model_name]["ro_an"])
        ro_un.append(reslults_dict[model_name]["ro_un"])
    
    results=[]
    for list_ in [ba_an,ba_un,no_an,no_un,ro_an,ro_un]:
        results_=[]
        for value1,value2,value3 in zip(list_[1],list_[2],list_[3]):
            results_.append((0+value1+value2+value3)>=2)
        results.append(results_)
    
    accuracy=[]
    for list_ in results:
        accuracy.append(sum(list_)/len(list_))
    
    print(accuracy)

if __name__ == "__main__":
    ablation_kind = "given_name"
    benchmark_version = "benchmark_v2"
    profile_version = "profile_v1"
    system_version = "system_v1"

    prompt_name = "prompt1"

    # model_list = ["gpt-3.5-turbo-16k", "gpt-4", "chatglm2-6b-32k", "chatglm2-6b", "XVERSE-13B-Chat", "Qwen-7B-Chat",
    #               "Qwen-14B-Chat", "vicuna-7b-v1.5-16k", "longchat-7b-16k", "longchat-13b-16k", "longchat-7b-32k-v1.5",
    #               "vicuna-13b-v1.5-16k"]
    model_list = ["gpt-3.5-turbo-1106","gpt-3.5-turbo-1106_tem_0.1","gpt-3.5-turbo-1106_tem_0.2","gpt-3.5-turbo-1106_tem_0.3"]
    prompt_kind = [ "few_shot"]
    person_name="homer"
    results={}
    for prompt in tqdm(prompt_kind):
        for model_name in model_list:
            ba_an,ba_un,no_an,no_un,ro_an,ro_un = Statistic(
                person_name=person_name,
                model_name=model_name,
                prompt_kind=prompt,
                prompt_name=prompt_name,
                benchmark_version=benchmark_version,
                profile_version=profile_version,
                system_version=system_version,
            ).show_to_list_true_false()
            results[model_name]={
                "ba_an":ba_an,
                "ba_un":ba_un,
                "no_an":no_an,
                "no_un":no_un,
                "ro_an":ro_an,
                "ro_un":ro_un
            }

    print(results)
    print(accuracy_single(results))
    print(accuracy_self_consistency(results))
        
