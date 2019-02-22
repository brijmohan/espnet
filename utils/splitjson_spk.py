#!/usr/bin/env python2
# encoding: utf-8

'''
splits json file based on speakers
'''

from __future__ import print_function
from __future__ import division

import argparse
import json
import logging
import os
import sys

import numpy as np
import itertools

is_python2 = sys.version_info[0] == 2

def flatten(l):
    return sorted(list(itertools.chain.from_iterable(l)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str,
                        help='json file')
    parser.add_argument('--dev', '-d', type=int,
                        help='Number of utterances to be assigned for dev',
                        default=4)
    parser.add_argument('--test', '-t', type=int,
                        help='Number of utterances to be assigned for test',
                        default=6)
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # check directory
    filename = os.path.basename(args.json).split('.')[0]
    dirname = os.path.dirname(args.json)
    dirname = '{}/split_utt_spk'.format(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # load json and split keys
    j = json.load(open(args.json))
    utt_ids = j['utts'].keys()
    logging.info("number of utterances = %d" % len(utt_ids))
    spk2utt = {}
    for utt_id, utt_obj in j['utts'].items():
        spkid = utt_obj['utt2spk']
        if spkid not in spk2utt:
            spk2utt[spkid] = []
        spk2utt[spkid].append(utt_id)
    logging.info("number of speakers = %d", len(spk2utt))
    num_utt = [len(x) for x in spk2utt.values()]
    logging.info("minimum number of utt = %d", min(num_utt))
    logging.info("maximum number of utt = %d", max(num_utt))

    dev_uttids = flatten([x[-args.dev:] for x in spk2utt.values()])
    test_uttids = flatten([x[-args.dev-args.test:-args.dev] for x in spk2utt.values()])
    train_uttids = flatten([x[:-args.dev-args.test] for x in spk2utt.values()])
    logging.info("Train = %d, Test = %d, Dev = %d", len(train_uttids), len(test_uttids), len(dev_uttids))


    with open('{}/{}.{}.json'.format(dirname, filename, 'train'), "wb+") as f:
        new_dic = {}
        for utt_id in train_uttids:
            new_dic[utt_id] = j['utts'][utt_id]
        jsonstr = json.dumps({'utts': new_dic},
                              indent=4,
                              ensure_ascii=False,
                              sort_keys=True)
        f.write(jsonstr.encode('utf_8'))

    with open('{}/{}.{}.json'.format(dirname, filename, 'test'), "wb+") as f:
        new_dic = {}
        for utt_id in test_uttids:
            new_dic[utt_id] = j['utts'][utt_id]
        jsonstr = json.dumps({'utts': new_dic},
                              indent=4,
                              ensure_ascii=False,
                              sort_keys=True)
        f.write(jsonstr.encode('utf_8'))

    with open('{}/{}.{}.json'.format(dirname, filename, 'dev'), "wb+") as f:
        new_dic = {}
        for utt_id in dev_uttids:
            new_dic[utt_id] = j['utts'][utt_id]
        jsonstr = json.dumps({'utts': new_dic},
                              indent=4,
                              ensure_ascii=False,
                              sort_keys=True)
        f.write(jsonstr.encode('utf_8'))

    '''
    if len(utt_ids) < args.parts:
        logging.error("#utterances < #splits. Use smaller split number.")
        sys.exit(1)
    utt_id_lists = np.array_split(utt_ids, args.parts)
    utt_id_lists = [utt_id_list.tolist() for utt_id_list in utt_id_lists]

    for i, utt_id_list in enumerate(utt_id_lists):
        new_dic = dict()
        for utt_id in utt_id_list:
            new_dic[utt_id] = j['utts'][utt_id]
        jsonstring = json.dumps({'utts': new_dic},
                                indent=4,
                                ensure_ascii=False,
                                sort_keys=True)
        fl = '{}/{}.{}.json'.format(dirname, filename, i + 1)
        if is_python2:
            sys.stdout = open(fl, "wb+")
            print(jsonstring.encode('utf_8'))
        else:
            sys.stdout = open(fl, "w+")
            print(jsonstring)
        sys.stdout.close()
    '''
