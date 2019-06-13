#!/bin/bash

dataset=train_plda_combined

local/gen_erep.sh --data-in /home/bsrivast/asr_tools/kaldi/egs/librispeech_spkv/v2/data/${dataset} \
	--data-out /home/bsrivast/asr_tools/kaldi/egs/librispeech_spkv/v2/data/kadv2_${dataset} \
	--modeldir /home/bsrivast/asr_tools/espnet/egs/librispeech_adv/asr1/exp/train_100_pytorch_adv_units512_layers3_grlalpha2/results \
	--erep-dir /home/bsrivast/asr_tools/kaldi/egs/librispeech_spkv/v2/exp/kadv2_${dataset}
