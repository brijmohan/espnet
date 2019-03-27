#!/bin/bash

dataset=train_100

local/gen_erep.sh --data-in /home/bsrivast/asr_tools/espnet/egs/librispeech_adv/asr1/data/${dataset} \
	--data-out /home/bsrivast/asr_tools/espnet/egs/librispeech_adv/asr1/data/erep_${dataset} \
	--modeldir exp/train_960_pytorch_vggblstm_e5_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs20_mli800_mlo150/results \
	--erep-dir /home/bsrivast/asr_tools/espnet/egs/librispeech_adv/asr1/exp/erep_${dataset}
