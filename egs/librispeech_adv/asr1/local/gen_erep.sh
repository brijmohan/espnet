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
nj=5
train_set=data/train_960
fbankdir=`pwd`/fbank
# feature configuration
do_delta=false
stage=1

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

set -e
set -u
set -o pipefail

feats_name=$(basename ${data_out})
feat_dir=dump/${feats_name}/delta${do_delta}

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

	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
	    ${data_out} exp/make_fbank/${feats_name} ${fbankdir}

	dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
		${data_out}/feats.scp ${train_set}/cmvn.ark exp/dump_feats/${feats_name} ${feat_dir}
	
fi

if [ ${stage} -le 1 ]; then

	# Split feats.scp for parallel processing
	split_dir=${feat_dir}/split${nj}feats

#	if [ -d ${split_dir} ]; then
#		rm -rf ${split_dir}
#	fi
#	mkdir -p ${split_dir}
#	sfeats=$(for n in `seq $nj`; do echo ${split_dir}/feats.$n.scp; done)
#	utils/split_scp.pl ${feat_dir}/feats.scp $sfeats
#
#	if [ -d ${erep_dir} ]; then
#	  rm -rf ${erep_dir}
#	fi
#	mkdir -p ${erep_dir}

	#### use CPU for decoding
	ngpu=1

	# set batchsize 0 to disable batch decoding
	#${decode_cmd} JOB=1:${nj} ${erep_dir}/log/erep.JOB.log \
	${decode_cmd} ${erep_dir}/log/erep.4.log \
	    asr_encode.py \
	    --ngpu ${ngpu} \
	    --batchsize 0 \
	    --feats-in ${split_dir}/feats.4.scp \
	    --feats-out ${erep_dir}/erep_${feats_name}.4 \
	    --model ${modeldir}/${recog_model}  \
	    &
	wait

	cat ${erep_dir}/*.scp > ${data_out}/erep_feats.scp
fi

echo "Finished"
