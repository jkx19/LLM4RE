import json
import numpy as np
from sklearn.metrics import f1_score

def get_gold(subset_len=500):
    f = open("test/test.json")
    testset = json.load(f)["request_states"][:subset_len]
    f.close()

    f = open("tacred/rel2id.json")
    rel2id = json.load(f)
    f.close()

    gold = []
    labels = []
    for idx, samp in enumerate(testset):
        txt = samp["instance"]["references"][0]["output"]["text"]
        if txt == "NA":
            # print(instance)
            label = "no_relation"
        else:
            label = txt.split(";")[1]
        labels.append(label.strip())
        gold.append(rel2id[label.strip()])

    return gold

def get_predict(fname:str):
    f = open(fname)
    lines = f.readlines()
    f.close()

    f = open("tacred/rel2id.json")
    rel2id = json.load(f)
    f.close()

    labelset = rel2id.keys()

    tuplelines = [""]

    for line in lines:
        if line.startswith("~~~~"):
            tuplelines.append("")
        else:
            line = line.strip() + " "
            tuplelines[-1] += line

    tuplelines.pop(-1)
    # print(tuplelines)

    predict = []

    for idx, raw in enumerate(tuplelines):
        raw = raw.replace("</s>", "")
        if raw == "":
            label = "no_relation"
        else:
            label = raw.split(" ")[0].lower().strip()
            while label[-1].isalpha() == False:
                label = label[:-1]
            # print(label)
            if label not in labelset:
                label = "no_relation"
        # print(label)
        predict.append(rel2id[label])

    return predict

if __name__ == "__main__":
    subset_len = 1000

    gold = get_gold(subset_len)
    predict = get_predict("answer/answer_1000.txt")

    gold = np.array(gold)
    predict = np.array(predict)
    labels = np.arange(42)

    micro_f1 = f1_score(gold, predict, labels=labels, average="micro")
    print(micro_f1)


# weighted_f1 = f1_score(gold, predict, labels=labels, average="weighted")


# f = open("no_tilde.txt", "w")
# for line in tuplelines:
#     s = line.find("(")
#     e = line.rfind(")")
#     if s == -1 or e == -1:
#         f.write("\n")
#     else:
#         f.write(line[s:e+1]+"\n")
# f.close()
# print(len(tuplelines))



# labels = set(labels)
# labels = list(labels)
# rel2id = {}
# rel2id["no_relation"] = 0
# idx = 1
# for lb in labels:
#     if lb == "no_relation":
#         continue
#     rel2id[lb] = idx
#     idx += 1

# f = open("test/test_map1000.json", "w")
# json.dump(rel2id)
        
