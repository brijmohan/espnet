
local/gen_erep.sh --data-in /home/bsrivast/kaldi/egs/librispeech_spkv/v2/data/train_960_combined_no_sil \
	--data-out /home/bsrivast/kaldi/egs/librispeech_spkv/v2/data/erep_train_960_combined_no_sil \
	--modeldir exp/pretrained_e2e_train_960 \
	--erep-dir /home/bsrivast/kaldi/egs/librispeech_spkv/v2/exp/erep_train_960_combined_no_sil
