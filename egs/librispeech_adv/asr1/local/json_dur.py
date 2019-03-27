'''
Get the combined duration of all utterances in the data json file
python json_dur.py <json file> <utt2dur file>
'''

import json
import sys

args = sys.argv
jf = args[1]
uf = args[2]

utt2dur = {}
with open(uf) as f:
    lines = f.read().splitlines()
    for line in lines:
        sp = line.split()
        utt2dur[sp[0]] = float(sp[1])

with open(jf) as f:
    j = json.loads(f.read())
    uttids = j['utts'].keys()

dur = sum([utt2dur[x] for x in uttids]) / 3600.0
print(dur)
