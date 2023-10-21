import requests
import json
from tqdm import tqdm
import re

def extract():
    sents = json.load(open("train/train_sent.json"))
    result = []
    gp = []
    for idx, sent in tqdm(enumerate(sents), total=len(sents)):
        gp.append(sent)
        if len(gp) < 20 and idx != len(sents)-1:
            continue
        req = requests.post("http://127.0.0.1:7171/oie_sents", json={"sentences": gp})
        tups = req.json()["triples"]
        result += tups
        gp = []

    print(len(result))
    f = open("openie/train.json", "w")
    json.dump(result, f)
    f.close()

    sents = json.load(open("test/test_sent.json"))
    result = []
    gp = []
    for idx, sent in tqdm(enumerate(sents), total=len(sents)):
        gp.append(sent)
        if len(gp) < 100 and idx != len(sents)-1:
            continue
        req = requests.post("http://127.0.0.1:7171/oie_sents", json={"sentences": gp})
        # req = requests.post("http://127.0.0.1:7171/oie_sent", json={"sentence": sent})
        tups = req.json()["triples"]
        result += tups
        # result.append(tups)
        gp = []

    f = open("openie/test.json", "w")
    json.dump(result, f)
    f.close()


def entities(sentence):
    entities = re.findall(r"<entity>(.*?)</entity>", sentence)
    entities = [x.strip() for x in entities]
    # print(entities)
    return entities

def maskentity(tups: list[list[str]], ent1, ent2):
    filtered = []
    for tup in tups:
        newtup = []
        exist = False
        for argument in tup:
            if ent1 in argument or ent2 in argument:
                exist = True
            argument = argument.replace(ent1, "ENTITY1")
            argument = argument.replace(ent2, "ENTITY2")
            newtup.append(argument)
        if exist:
            filtered.append(newtup)
    return filtered

def processtuple():

    testraw = json.load(open("openie/test.json"))
    testset = json.load(open("test/test_labeled.json"))
    print(len(testraw))
    result = []
    for idx, ext in enumerate(testraw):
        sent = testset[idx]["input"]
        ents = entities(sent)
        print(ents)
        tups = []
        for raw in ext:
            tuplist = re.findall("\(.*\)", raw)
            if len(tuplist) == 0:
                continue
            tupstr = tuplist[0][1:-1]
            tup = tupstr.split(";")
            tup = [x.strip() for x in tup]
            tups.append(tup)
        tups = maskentity(tups, ents[0], ents[1])    
        # print(tups)
        result.append(tups)

    f = open("openie/test_processed.json", "w")
    json.dump(result, f)
    f.close()

    testraw = json.load(open("openie/train.json"))
    testset = json.load(open("tacred/train_labeled.json"))
    print(len(testraw))
    result = []
    for idx, ext in enumerate(testraw):
        sent = testset[idx]["input"]
        ents = entities(sent)
        # print(ents)
        tups = []
        for raw in ext:
            tuplist = re.findall("\(.*\)", raw)
            if len(tuplist) == 0:
                continue
            tupstr = tuplist[0][1:-1]
            tup = tupstr.split(";")
            tup = [x.strip() for x in tup]
            tups.append(tup)
        tups = maskentity(tups, ents[0], ents[1])    
        # print(tups)
        result.append(tups)

    f = open("openie/train_processed.json", "w")
    json.dump(result, f)
    f.close()


# import numpy as np
import sys
sys.path.append("/data/jkx/Documents/Relation/CaRB")
from CaRB.carb import Benchmark
from CaRB.oie_readers.extraction import Extraction
from CaRB.matcher import Matcher
# from operator import itemgetter
import random
    
random.seed(42)
for sample in [20000]: 

    testtups = json.load(open("openie/test_processed.json"))
    traintups = json.load(open("openie/train_processed.json"))
    testsent = json.load(open("test/test_sent.json"))
    trainsent = json.load(open("train/train_sent.json"))

    sample_idx = random.sample(range(len(trainsent)), sample)

    stups, ssent = [], []
    for i in sample_idx:
        stups.append(traintups[i])
        ssent.append(trainsent[i])
    traintups = stups
    trainsent = ssent

    total = 0
    mutual = []
    for testidx, testtup in tqdm(enumerate(testtups), total=len(testtups)):
        dist = []
        for trainidx, traintup in enumerate(traintups):
            testexts = []
            for tup in testtup:
                try:
                    pred = Extraction(pred=tup[0], head_pred_index=-1, sent=testsent[testidx], confidence=-1)
                except:
                    continue
                for i in range(1, len(tup)):
                    pred.addArg(tup[i])
                testexts.append(pred)
            trainexts = []
            for tup in traintup:
                try:
                    pred = Extraction(pred=tup[0], head_pred_index=-1, sent=testsent[testidx], confidence=-1)
                except:
                    continue
                for i in range(1, len(tup)):
                    pred.addArg(tup[i])
                trainexts.append(pred)
            
            # print(testsent[testidx])
            testpred = {testsent[testidx]: testexts}
            trainpred = {testsent[testidx]: trainexts}

            bench = Benchmark()
            bench.gold = testpred
            auc, [p, r, f1] = bench.compare(trainpred, Matcher.binary_linient_tuple_match)
            # print(auc, p, r, f1)
            if auc > 0 or f1 > 0:
                # print(traintup, testtup)
                dist.append([auc, sample_idx[trainidx]])
        if len(dist) > 0:
            total += 1
        mutual.append(dist)
            
    print(total)
    f = open(f"mutual/openie_{sample}.json", "w")
    json.dump(mutual, f)
    f.close()
        

