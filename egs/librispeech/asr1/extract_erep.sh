#!/bin/bash

dataset=test_clean_trial

local/gen_erep.sh --data-in /home/bsrivast/kaldi/egs/librispeech_spkv/v2/data/${dataset} \
	--data-out /home/bsrivast/kaldi/egs/librispeech_spkv/v2/data/erep_${dataset} \
	--modeldir exp/pretrained_e2e_train_960 \
	--erep-dir /home/bsrivast/kaldi/egs/librispeech_spkv/v2/exp/erep_${dataset}
