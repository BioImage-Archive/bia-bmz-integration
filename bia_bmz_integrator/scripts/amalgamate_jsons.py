import json
import glob

def amalgamate_jsons():
    
    all_results = []
    for file in glob.glob('./results/jsons/*.json'):
        with open(file) as f:
            all_results.extend(json.load(f))

    with open('./results/all_results.json', 'w') as out_f:
        json.dump(all_results, out_f, indent=4)