#!/bin/bash

# Usage:
# local/gen_erep.sh --data-in <input-data-dir> --data-out <output-data-dir> --modeldir <path/to/model> --erep-dir <output-feats-dump-dir>

. ./path.sh
. ./cmd.sh

data_in=
data_out=
modeldir=
erep_dir=

recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
nj=3
train_set=data/train_960
fbankdir=`pwd`/fbank
# feature configuration
do_delta=false
stage=0

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

set -e
set -u
set -o pipefail

feats_name=$(basename ${data_out})
feat_dir=dump/${feats_name}/delta${do_delta}
vadfeats=/media/data/vadfeats

#mkdir -p ${vadfeats}

if [ ${stage} -le 0 ]; then

	if [ -d ${data_out} ]; then
	  rm -rf ${data_out}
	fi
	cp -r ${data_in} ${data_out}

	# Make feats compatible with E2E model
	if [ -d ${feat_dir} ]; then
	  rm -rf ${feat_dir}
	fi
	mkdir -p ${feat_dir}

	: '
	# Compute fbank with energy
	steps/make_fbank_pitch.sh --fbank-config conf/fbank_energy.conf --cmd "$train_cmd" --nj 29 --write_utt2num_frames true \
	    ${data_out} exp/make_fbank/${feats_name} ${fbankdir}

	# Compute VAD
	sid/compute_vad_decision.sh --nj 29 --cmd "$train_cmd" \
		${data_out} exp/make_vad $vadfeats
	utils/fix_data_dir.sh ${data_out}

	# Store VAD somewhere else
	cp ${data_out}/vad.scp local/
	
	# Make fbank without energy for ASR decoding
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 29 --write_utt2num_frames true \
	    ${data_out} exp/make_fbank/${feats_name} ${fbankdir}
	# Copy back the VAD
	mv local/vad.scp ${data_out}/


	# Apply VAD and select frames
	apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${data_out}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${data_out}/vad.scp ark,scp:${vadfeats}/${feats_name}.ark,${vadfeats}/${feats_name}.scp

	# Use new feats for xvectors
	cp ${vadfeats}/${feats_name}.scp ${data_out}/feats.scp
	rm ${data_out}/vad.scp
	'
	# Make fbank without energy for ASR decoding
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 40 --write_utt2num_frames true \
	    ${data_out} exp/make_fbank/${feats_name} ${fbankdir}
	utils/fix_data_dir.sh ${data_out}
	if [ -f ${data_out}/vad.scp ]; then
	  rm ${data_out}/vad.scp
	fi

	dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
		${data_out}/feats.scp ${train_set}/cmvn.ark exp/dump_feats/${feats_name} ${feat_dir}
	
fi

if [ ${stage} -le 1 ]; then

	# Split feats.scp for parallel processing
	split_dir=${feat_dir}/split${nj}feats
	if [ -d ${split_dir} ]; then
		rm -rf ${split_dir}
	fi
	mkdir -p ${split_dir}
	sfeats=$(for n in `seq $nj`; do echo ${split_dir}/feats.$n.scp; done)
	utils/split_scp.pl ${feat_dir}/feats.scp $sfeats

	if [ -d ${erep_dir} ]; then
	  rm -rf ${erep_dir}
	fi
	mkdir -p ${erep_dir}

	#### use CPU for decoding
	ngpu=1

	# set batchsize 0 to disable batch decoding
	#${decode_cmd} ${erep_dir}/log/erep.3.log \
	${decode_cmd} JOB=1:${nj} ${erep_dir}/log/erep.JOB.log \
	    asr_encode.py \
	    --ngpu ${ngpu} \
	    --batchsize 0 \
	    --feats-in ${split_dir}/feats.JOB.scp \
	    --feats-out ${erep_dir}/erep_${feats_name}.JOB \
	    --model ${modeldir}/${recog_model}  \
	    &
	wait

	mv ${data_out}/feats.scp ${data_out}/fbank_feats.scp
	cat ${erep_dir}/*.scp > ${data_out}/feats.scp
fi

echo "Finished"
