import requests
import json
import sys
from tqdm import tqdm

# req = requests
# reg = requests.post("http://127.0.0.1:7153/prompt_llama2", json={'input': "Hi, how are you?"})
# print(reg.text)
# exit()

def ask_llama(sample):
    f = open(f"prompt/prompt_oie_{sample}.json", "r", encoding="utf-8")
    contents = json.load(f)
    f.close()
    output = []

    for idx, stuff in tqdm(enumerate(contents), total=len(contents)):
        msg = stuff
        result = ""
        # print(stuff)
        # exit()
        # try:
            # print(line)
        response = requests.post(
            url="http://127.0.0.1:7153/prompt_llama2",
            json={"input": msg, "temperature": 0.05}
        )
        if response.text == "Internal Server Error":        
            output.append("\n")
            output.append("~~~~~~~~~~~~")
            continue
        
        # print(response.text)
        res = json.loads(response.text)["response"]
        result = res[len(msg):].strip()
        # print(result)
        # exit()
        output.append(result)
        output.append("~~~~~~~~~~~~")

        # if idx > 8:
        #     break
    f= open(f"answer/answer_{sample}.txt", "w", encoding="utf-8")
    for line in output:
        f.write(line+"\n")
    f.close()


if __name__ == "__main__":
    for sample in [100, 500, 1000, 2000, 5000, 20000, 60000]:
        ask_llama(sample)
