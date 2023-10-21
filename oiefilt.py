import json
# import numpy as np
import random
from tqdm import tqdm


INSTRUCTION = "Please classify relationships between the two entities (marked with <entity> and </entity>). The set of relationships is as follows: [org:founded, org:subsidiaries, per:date_of_birth, per:cause_of_death, per:age, per:stateorprovince_of_birth, per:countries_of_residence, per:country_of_birth, per:stateorprovinces_of_residence, org:website, per:cities_of_residence, per:parents, per:employee_of, no_relation, per:city_of_birth, org:parents, org:political/religious_affiliation, per:schools_attended, per:country_of_death, per:children, org:top_members/employees, per:date_of_death, org:members, org:alternate_names, per:religion, org:member_of, org:city_of_headquarters, per:origin, org:shareholders, per:charges, per:title, org:number_of_employees/members, org:dissolved, org:country_of_headquarters, per:alternate_names, per:siblings, org:stateorprovince_of_headquarters, per:spouse, per:other_family, per:city_of_death, per:stateorprovince_of_death, org:founded_by]. Here are some examples. "


def complete(demolist:list, shot):
    # print(demolist)
    demoset = set(demolist)
    while len(demoset) < shot:
        demoset.add(random.randint(1, 60000))
    return list(demoset)
            

def find_idx(sample=1000, shot=30):
    f = open(f"mutual/openie_{sample}.json")
    distances = json.load(f)
    f.close()

    demo_set = []

    random.seed(42)


    for sidx, disttup in tqdm(enumerate(distances), total=len(distances)):
        disttup.sort(key=lambda a: a[0])
        bestlist = disttup[-shot:]
        # exit()
        bestlist = [a[1] for a in bestlist]
        if len(bestlist) < shot:
            bestlist = complete(bestlist, shot)
        demo_set.append(bestlist)
        
    f = open(f"mutual/demo_oie_{sample}.json", "w")
    json.dump(demo_set, f)
    f.close()


def buildprompt_llama(sample):
    train_data = json.load(open("tacred/train_labeled.json"))
    examples = []
    for instance in train_data:
        examples.append({
            "input": instance["input"],
            "output": instance["label"]
        })

    f = open(f"mutual/demo_oie_{sample}.json")
    demos_idx = json.load(f)
    f.close()

    test_json = json.load(open("test/test.json"))
    # prompt = test_json["prompt"]
    # test_list = test_json["request_states"]
    test_sentences = [x["instance"]["input"]["text"] for x in test_json["request_states"]][:1000]

    messages = []

    for idx, data in enumerate(test_sentences):
        message = INSTRUCTION
        demos = [examples[j] for j in demos_idx[idx]]
        random.shuffle(demos)
        for demo in demos:
            sent, answer  = demo['input'], demo["output"]
            message += f"Sentence: {sent} Output: {answer}. "
        message += f"Now please classify the relation of the two entities in the following Sentence: {data} Output:"
        messages.append(message)

    f = open(f"prompt/prompt_oie_{sample}.json", "w")
    json.dump(messages, f)
    f.close()
    print(len(test_sentences))


def build_prompt(sample):
    train_data = json.load(open("tacred/train_labeled.json"))
    examples = []
    for instance in train_data:
        examples.append({
            "input": instance["input"],
            "output": instance["label"]
        })

    f = open(f"mutual/demo_oie_{sample}.json")
    demos_idx = json.load(f)
    f.close()

    test_json = json.load(open("test/test.json"))
    # prompt = test_json["prompt"]
    # test_list = test_json["request_states"]
    test_sentences = [x["instance"]["input"]["text"] for x in test_json["request_states"]]

    messages = []
    for idx, data in enumerate(test_sentences):
        message = [
            {"role": "system", "content": "You are a helpful, pattern-following assistant."}
        ]
        message.append({
            "role": "user",
            "content": INSTRUCTION
        })
        demos = [examples[j] for j in demos_idx[idx]]
        random.shuffle(demos)
        for demo in demos:
            message.append({
                "role": "user", "content": "Text: " + demo["input"] + "\n"
            })
            message.append({
                "role": "assistant","content": demo["output"] + "\n"
            })
        message.append({"role": "user", "content": data})
        messages.append(message)

    f = open(f"prompt/prompt_oie_{sample}.json", "w")
    json.dump(messages, f)
    f.close()

if __name__ == "__main__":
    for sample in [100, 500, 1000, 2000, 5000, 20000, 60000]:
    # for sample in [100, 500, 1000]:
        find_idx(sample=sample, shot=20)
        buildprompt_llama(sample)