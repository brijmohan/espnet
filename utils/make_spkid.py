#!/usr/bin/env python2
# encoding: utf-8


import argparse
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('utt2spk', type=str, help='utt2spk file')
    parser.add_argument('spkid', type=str, help='spkid output file')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.info("reading %s", args.utt2spk)
    with open(args.utt2spk, 'r') as f1, open(args.spkid, 'w') as f2:
        lines = f1.read().splitlines()
        spks = sorted(set([x.split()[1] for x in lines]))
        logging.info("Number of speakers: %d", len(spks))
        for i, s in enumerate(spks):
            f2.write(s + ' ' + str(i) + '\n')
