from GPTMan.benchmark.benchmark_file_util import \
    QUESTIONNAIRE_PATH_ROLES_RELATION_Q, \
    QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q, \
    QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q, \
    load_json_file, \
    COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q, \
    COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q, \
    COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q


def count_rates_answer_able(path, person_name):
    obj = load_json_file(path.format(person_name=person_name))

    unanswerable = 0

    for count in range(len(obj)):
        question_odj = obj[count]
        if question_odj['gold_answer'].lower() == "i don't know":
            unanswerable += 1

    return unanswerable / len(obj), (len(obj) - unanswerable) / len(obj), len(obj)


def count_accuracy(path, person_name, model_name):
    obj = load_json_file(path.format(person_name=person_name))
    answerable_list = []
    unanswerable_list = []

    for count in range(len(obj)):
        question_odj = obj[count]
        if question_odj['gold_answer'].lower() == "i don't know":
            unanswerable_list.append(question_odj)
        else:
            answerable_list.append(question_odj)

    correct_answerable = 0
    correct_unanswerable = 0

    for question_odj in answerable_list:

        if question_odj['gold_answer'].lower() == question_odj[f'generated_answer_{model_name}'].lower():
            correct_answerable += 1
        elif question_odj[f'generated_answer_{model_name}'].lower() in question_odj['gold_answer'].lower():
            correct_answerable += 1

    for question_odj in unanswerable_list:

        if question_odj['gold_answer'].lower() == question_odj[f'generated_answer_{model_name}'].lower():
            correct_unanswerable += 1
        elif question_odj[f'generated_answer_{model_name}'].lower() in question_odj['gold_answer'].lower():
            correct_unanswerable += 1

    """if len(answerable_list)==0:
        return 0, correct_unanswerable/len(unanswerable_list)
    elif len(unanswerable_list)==0:
        return correct_answerable/len(answerable_list), 0
    else:"""
    return correct_answerable/len(answerable_list),correct_answerable,len(answerable_list),\
        correct_unanswerable/len(unanswerable_list),correct_unanswerable,len(unanswerable_list)

if __name__ == "__main__":
    print(count_accuracy(QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q, 'monica','gpt-3.5-turbo-16k'))
    #print(count_accuracy(QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q, 'monica','gpt-4'))
    #print(count_accuracy(QUESTIONNAIRE_PATH_ROLES_RELATION_Q, 'monica','gpt-3.5-turbo-16k'))
    print(count_accuracy(QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q, 'monica', 'gpt-4'))
    #print(count_accuracy(QUESTIONNAIRE_PATH_ROLES_RELATION_Q, 'monica', 'gpt-4'))

