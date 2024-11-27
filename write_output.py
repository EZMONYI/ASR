import json
import re
import nltk
import os

def get_results():
    results = []
    for filename in os.listdir('output/failed'):
        file_path = os.path.join('output/failed', filename)
        with open(file_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            audio = data['path']
            text = data['response'].split(':')[-1]
            results.append([audio, text])
    return results




def get_test_data():
    data_path = 'data/test_data_1126/'
    cantonese = []
    mandarin = []
    with open(f"{data_path}test_yue_phoenix_tv_data.list", 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            key = data['key']
            wav = data['wav']
            txt = data['txt']
            wav_file = data_path + 'phoenix_tv/' + wav.split('/')[-1]
            cantonese.append([wav_file, txt])
    with open(f"{data_path}test_zh_aishell4_data.list", 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            key = data['key']
            wav = data['wav']
            txt = data['txt']
            wav_file = data_path + 'aishell4/' + wav.split('/')[-1]
            mandarin.append([wav_file, txt])
    return cantonese, mandarin


def write_json(truth, inferences, lang):
    truth_table = {}
    for pair in truth:
        audio, text = pair
        audio = audio.split('/')[-1]
        truth_table[audio] = text
    results = []
    for pair in inferences:
        audio, text = pair
        audio = audio.split('/')[-1]
        text = re.sub(r'["\']', '', text.strip())
        try:
            truth = truth_table[audio]
        except:
            continue
        results.append({'wav': audio, 'truth': truth, 'inference': text})
    with open(f"output/{lang}_result.json", 'w+', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4)
    for r in results:
        with open(f'output/truth_{lang}.txt', 'a') as f:
            f.write(f"{r['wav']} {r['truth']}\n")
    for r in results:
        with open(f'output/inference_{lang}.txt', 'a') as f:
            f.write(f"{r['wav']} {r['inference']}\n")
    

if __name__ == "__main__":
    cantonese, mandarin = get_test_data()
    inferences = get_results()
    write_json(cantonese, inferences, 'cantonese')
    write_json(mandarin, inferences, 'mandarin')
