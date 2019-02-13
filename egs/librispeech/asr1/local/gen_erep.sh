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
nj=10
train_set=data/train_960
fbankdir=`pwd`/fbank
# feature configuration
do_delta=false

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

set -e
set -u
set -o pipefail


feats_name=$(basename ${data_out})

if [ -d ${data_out} ]; then
  rm -rf ${data_out}
fi
cp -r ${data_in} ${data_out}

# Make feats compatible with E2E model
feat_dir=dump/${feats_name}/delta${do_delta}; mkdir -p ${feat_dir}

steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
    ${data_out} exp/make_fbank/${feats_name} ${fbankdir}

dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
	${data_out}/feats.scp ${train_set}/cmvn.ark exp/dump_feats/${feats_name} ${feat_dir}


# Split feats.scp for parallel processing
mkdir -p ${feat_dir}/split${nj}feats
mkdir -p ${erep_dir}
sfeats=$(for n in `seq $nj`; do echo ${feat_dir}/split${nj}feats/feats.$n.scp; done)
utils/split_scp.pl ${feat_dir}/feats.scp $sfeats

#### use CPU for decoding
ngpu=0

# set batchsize 0 to disable batch decoding
${decode_cmd} JOB=1:${nj} ${erep_dir}/log/erep.JOB.log \
    asr_encode.py \
    --ngpu ${ngpu} \
    --batchsize 0 \
    --feats-in ${feat_dir}/split${nj}feats/feats.JOB.scp \
    --feats-out ${erep_dir}/erep_${feats_name}.JOB \
    --model ${modeldir}/${recog_model}  \
    &
wait

cat ${erep_dir}/*.scp > ${data_out}/erep_feats.scp

echo "Finished"
