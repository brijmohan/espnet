#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os
import re

# chainer related
import chainer

from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

# torch related
import torch

# espnet related
from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import load_inputs_and_targets
from espnet.asr.asr_utils import make_batchset
from espnet.asr.asr_utils import PlotAttentionReport
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import torch_snapshot
from espnet.nets.e2e_asr_th import E2E
from espnet.nets.e2e_asr_th import Loss
from espnet.nets.e2e_asr_th import pad_list

# for kaldi io
import kaldi_io_py

# rnnlm
import espnet.lm.extlm_pytorch as extlm_pytorch
import espnet.lm.lm_pytorch as lm_pytorch

# matplotlib related
import matplotlib
import numpy as np
matplotlib.use('Agg')

REPORT_INTERVAL = 100


def get_alpha_number(s):
    m = re.search(r'\d+$', s)
    m1 = int(m.group()) if m else None
    t = re.search(r'^[a-z]+', s)
    t1 = t.group() if t else None
    return (t1, m1)

def get_advsched(advstr, nepochs):
    advsched = {}
    sp = [get_alpha_number(x) for x in advstr.split(',')]
    assert sum([x[1] for x in sp]) == nepochs, "Sum of schedule segment != nepochs"
    ecnt = 0
    for m, t in sp:
        for i in range(t):
            advsched[ecnt] = m
            ecnt += 1

    # Hack to prevent KeyError in last epoch, add last mode
    advsched[ecnt] = sp[-1][0]
    return advsched

def get_grlalpha(max_grlalpha, ep_num, total_epochs):
    p_i = ep_num / float(total_epochs)
    cga = float(max_grlalpha * (2.0 / (1.0 + np.exp(-10 * p_i)) - 1.0))
    logging.info(" ------------- CGA = %f ---------", cga)
    return cga

class CustomEvaluator(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    x = self.converter(batch, self.device)
                    self.model(*x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu, adv_schedule=None,
                 max_grlalpha=None):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.adv_schedule = adv_schedule
        self.last_adv_mode = None
        self.max_grlalpha = max_grlalpha

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        x = self.converter(batch, self.device)

        curr_epoch = int(self.epoch_detail)
        adv_mode = self.adv_schedule[curr_epoch]
        logging.info("Epoch detail = %f, Adv mode = %s", self.epoch_detail,
                     adv_mode)
        # If transitioning to speaker branch training - RESET!
        if curr_epoch > 0:
            if self.last_adv_mode != 'spk' and adv_mode == 'spk':
                logging.info(" ----- Resetting the adversarial branch weights... -----")
                if self.ngpu > 1:
                    self.model.module.predictor.adv.init_like_chainer()
                else:
                    self.model.predictor.adv.init_like_chainer()

                    #logging.info("Some weights after resetting ---")
                    #for p in self.model.predictor.adv.advnet.parameters():
                    #    logging.info(p)
                    #    break

        # UNCOMMENT NEXT LINE TO ALLOW exponentially growing alpha
        #curr_grlalpha = get_grlalpha(self.max_grlalpha, self.epoch_detail,
        #                             len(self.adv_schedule))
        curr_grlalpha = self.max_grlalpha

        loss_asr, loss_adv = self.model(*x, grlalpha=curr_grlalpha)

        if adv_mode == 'spk':
            if self.ngpu > 1:
                self.model.module.predictor.freeze_encoder()
            else:
                self.model.predictor.freeze_encoder()

            loss = loss_adv
        elif adv_mode == 'asr':
            if self.ngpu > 1:
                self.model.module.predictor.freeze_encoder()
            else:
                self.model.predictor.freeze_encoder()
            loss = loss_asr
        else:
            if self.ngpu > 1:
                self.model.module.predictor.unfreeze_encoder()
            else:
                self.model.predictor.unfreeze_encoder()
            loss = loss_asr + loss_adv

        # Compute the loss at this time step and accumulate it
        optimizer.zero_grad()  # Clear the parameter gradients
        if self.ngpu > 1:
            loss = 1. / self.ngpu * loss
            loss.backward(loss.new_ones(self.ngpu))  # Backprop
        else:
            loss.backward()  # Backprop
        loss.detach_()  # Truncate the graph
        loss_asr.detach_()
        loss_adv.detach_()
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()

        # Update last adv mode
        self.last_adv_mode = adv_mode


class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, subsamping_factor=1):
        self.subsamping_factor = subsamping_factor
        self.ignore_id = -1

    def transform(self, item):
        return load_inputs_and_targets(item)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        xs, ys, y_adv = batch[0]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)
        y_adv_pad = pad_list([torch.from_numpy(y).long() for y in y_adv], 0).to(device)

        return xs_pad, ilens, ys_pad, y_adv_pad


def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))
    odim_adv = None
    if args.adv:
        odim_adv = int(valid_json[utts[0]]['output'][1]['shape'][1])
        logging.info('#output dims adversarial: ' + str(odim_adv))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # specify model architecture
    e2e = E2E(idim, odim, args, odim_adv=odim_adv)
    model = Loss(e2e, args.mtlalpha)

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        e2e.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, odim_adv, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # Setup an optimizer
    # First distinguish between learning rates
    if args.ngpu > 1:
        param_grp = [
            {'params': model.module.predictor.enc.parameters(), 'lr': args.asr_lr},
            {'params': model.module.predictor.dec.parameters(), 'lr': args.asr_lr},
            {'params': model.module.predictor.adv.parameters(), 'lr': args.adv_lr}
        ]
    else:
        param_grp = [
            {'params': model.predictor.enc.parameters(), 'lr': args.asr_lr},
            {'params': model.predictor.dec.parameters(), 'lr': args.asr_lr},
            {'params': model.predictor.adv.parameters(), 'lr': args.adv_lr}
        ]
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(param_grp, rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(param_grp)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(e2e.subsample[0])

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(train, converter.transform),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
        valid_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = chainer.iterators.SerialIterator(
            TransformDataset(train, converter.transform),
            batch_size=1)
        valid_iter = chainer.iterators.SerialIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False)


    # Prepare adversarial training schedule dictionary
    adv_schedule = get_advsched(args.adv, args.epochs)

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer, converter, device,
        args.ngpu, adv_schedule=adv_schedule, max_grlalpha=args.grlalpha)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        #torch_resume(args.resume, trainer, weight_sharing=args.weight_sharing)
        torch_resume(args.resume, trainer, weight_sharing=args.weight_sharing,
                    reinit_adv=args.reinit_adv)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.predictor.calculate_all_attentions
        else:
            att_vis_fn = model.predictor.calculate_all_attentions
        trainer.extend(PlotAttentionReport(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, device=device), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_ctc', 'validation/main/loss_ctc',
                                          'main/loss_att',
                                          'validation/main/loss_att',
                                          'main/loss_adv',
                                          'validation/main/loss_adv'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc',
                                          'main/acc_adv',
                                          'validation/main/acc_adv'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode is not 'ctc':
        trainer.extend(extensions.snapshot_object(model, 'model.acc.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode is not 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(REPORT_INTERVAL, 'iteration'))
        report_keys.append('eps')
    if args.report_cer:
        report_keys.append('validation/main/cer')
    if args.report_wer:
        report_keys.append('validation/main/wer')
    if args.adv:
        report_keys.extend(['main/loss_adv', 'main/acc_adv',
                            'validation/main/loss_adv',
                            'validation/main/acc_adv'])
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    # Run the training
    trainer.run()


def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    idim, odim, odim_adv, train_args = get_model_conf(args.model, args.model_conf)

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, odim, train_args, odim_adv=odim_adv)
    model = Loss(e2e, train_args.mtlalpha)
    if train_args.rnnlm is not None:
        # set rnnlm. external rnnlm is used for recognition.
        model.predictor.rnnlm = rnnlm
    torch_load(args.model, model)
    e2e.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = range(args.ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
                nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
                new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)
    else:
        try:
            from itertools import zip_longest as zip_longest
        except Exception:
            from itertools import izip_longest as zip_longest

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data
        keys = list(js.keys())
        feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
        sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
        keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                feats = [kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
                         for name in names]
                nbest_hyps = e2e.recognize_batch(feats, args, train_args.char_list, rnnlm=rnnlm)
                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))


def encode(args):
    '''Get ASR encoded representations...probably for xvectors'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    idim, odim, odim_adv, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, odim, train_args, odim_adv=odim_adv)
    model = Loss(e2e, train_args.mtlalpha)
    if train_args.rnnlm is not None:
        # set rnnlm. external rnnlm is used for recognition.
        model.predictor.rnnlm = rnnlm
    torch_load(args.model, model)
    e2e.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = range(args.ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()

    arkscp = 'ark:| copy-feats --print-args=false ark:- ark,scp:%s.ark,%s.scp' % (args.feats_out, args.feats_out)

    if args.batchsize == 0:
        with torch.no_grad():
            with kaldi_io_py.open_or_fd(arkscp, 'wb') as f, open(args.feats_in, 'rb') as f2:
                lines = f2.read().splitlines()
                for idx, line in enumerate(lines, 1):
                    line = line.strip().split()
                    name = line[0]
                    logging.info('(%d/%d) decoding ' + name, idx, len(lines))
                    feat = kaldi_io_py.read_mat(line[1])
                    rep = e2e.erep(feat)
                    logging.info('Rep shape: %s', rep.shape)
                    kaldi_io_py.write_mat(f, rep, name)
    else:
        try:
            from itertools import zip_longest as zip_longest
        except Exception:
            from itertools import izip_longest as zip_longest

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # Create json object for batch processing
        logging.info("Creating json for batch processing...")
        js = {}
        with open(args.feats_in, 'rb') as f:
            lines = f.read().splitlines()
            for line in lines:
                line = line.strip().split()
                name = line[0]
                featpath = line[1]
                feat_shape = kaldi_io_py.read_mat(featpath).shape
                js[name] = { 'feat': featpath, 'shape': feat_shape }

        # sort data
        logging.info("Sorting data for batch processing...")
        keys = list(js.keys())
        feat_lens = [js[key]['shape'][0] for key in keys]
        sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
        keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
                for names in grouper(args.batchsize, keys, None):
                    names = [name for name in names if name]
                    feats = [kaldi_io_py.read_mat(js[name]['feat'])
                             for name in names]
                    reps, replens = e2e.erep_batch(feats)
                    print(reps.shape, replens)
                    for i, rep in enumerate(reps):
                        name = names[i]
                        kaldi_io_py.write_mat(f, rep, name)

