'''
Copies filterbank json and replaces feats and shape with new erep feats
'''

import os
from os.path import exists, join

import json

# for kaldi io
import kaldi_io_py

old_json_dir = 'dump/train_100/deltafalse/split_utt_spk'
new_feats_scp = 'data/erep_train_100/feats.scp'

new_json_dir = 'data/erep_train_100/json'

if not exists(new_json_dir):
    os.makedirs(new_json_dir)

print('Create feats dictionary...')
feats_dict = {}
with open(new_feats_scp) as f:
    for line in f.read().splitlines():
        sp = line.split()
        feats_dict[sp[0]] = {
                        u'path': sp[1], 
                        u'shape': kaldi_io_py.read_mat(sp[1]).shape
                    }
print('Done reading features!')

print('Reading data jsons...')
djsons = [x for x in os.listdir(old_json_dir) if x.endswith('.json')]
for jsfile in djsons:
    print('Reading ' + jsfile)
    with open(join(old_json_dir, jsfile)) as f1, open(join(new_json_dir,
                                                           jsfile), 'w') as f2:
        js = json.load(f1)
        utt_ids = js[u'utts'].keys()
        for k in utt_ids:
            js[u'utts'][k][u'input'][0][u'feat'] = feats_dict[k][u'path']
            js[u'utts'][k][u'input'][0][u'shape'] = feats_dict[k][u'shape']

        json.dump(js, f2, indent=4)

print('Done!')

