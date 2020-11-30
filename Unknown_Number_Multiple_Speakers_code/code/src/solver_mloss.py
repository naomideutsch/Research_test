import os
import time

import torch
import subprocess

from pit_criterion import cal_loss
import pdb
dbstop = pdb.set_trace


class Solver(object):

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']

        print(len(self.tr_loader))

        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        self.segment = args.segment
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)

        self._reset()
        self.data_dir = args.data_dir
        self.cal_sdr = args.cal_sdr
        self.batch_size_eval = args.batch_size_eval
        self.sample_rate = args.sample_rate
        self.eval_every = args.eval_every
        self.lr_decay = args.lr_decay
        self.args = args

    def _reset(self):
        # Reset
        if self.continue_from:

            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])

            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            print('Epoch - ' + str(epoch))
            if epoch % 2 == 0:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] * self.lr_decay
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))

            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

            if epoch % self.eval_every == 0:

                cmd = ['../wsj0/utils/run.pl', '--mem', '4G', '--gpu', '1', str(self.save_folder)+'/evaluate_'+str(epoch)+'.log', 'CUDA_VISIBLE_DEVICES=1', 'evaluate_mloss.py']
                cmd += ['--model_path', str(self.save_folder) + '/final.pth.tar']
                cmd += ['--data_dir', str(self.data_dir), '--cal_sdr' , str(self.cal_sdr)]
                cmd += ['--use_cuda', '1', '--sample_rate', str(self.sample_rate), '--batch_size', str(self.batch_size_eval)]
                print('cmd is - ' + str(cmd))
                subprocess.Popen(cmd)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source = data


            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            estimate_source, est_mask_f_all = self.model(padded_mixture)

            loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(padded_source, estimate_source, mixture_lengths)
            #
            # import librosa
            # import numpy as np
            # print('estimate_source ' + str(estimate_source.shape))
            # print('padded_source ' + str(padded_source.shape))
            #
            #
            # wav = padded_mixture[0,:].detach().float().cpu().numpy()
            # wav = wav / np.max(wav)
            # librosa.output.write_wav('/private/home/eliyan/mix.wav', y=wav, sr=8000)
            #
            #
            # wav = estimate_source[0,0,:].detach().float().cpu().numpy()
            # wav = wav / np.max(wav)
            # librosa.output.write_wav('/private/home/eliyan/gt.wav', y=wav, sr=8000)
            #
            # wav2 = padded_source[0,0,:].detach().cpu().numpy()
            # librosa.output.write_wav('/private/home/eliyan/pred.wav', y=wav2, sr=8000)
            #
            # wav = estimate_source[0,1,:].detach().float().cpu().numpy()
            # wav = wav / np.max(wav)
            # librosa.output.write_wav('/private/home/eliyan/gt1.wav', y=wav, sr=8000)
            #
            # wav2 = padded_source[0,1,:].detach().cpu().numpy()
            # librosa.output.write_wav('/private/home/eliyan/pred1.wav', y=wav2, sr=8000)
            # exit()


            loss_ss_f_all = 0.0
            iik = 0
            for ii in range(len(est_mask_f_all)):
                if (ii+1) % self.args.loss_every == 0:

                    loss_ss_f_ii = cal_loss(padded_source, est_mask_f_all[ii], mixture_lengths)
                    loss_ss_f_all += loss_ss_f_ii[0]
                    iik += 1

            loss += loss_ss_f_all
            loss /= (iik+1)


            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')


        return total_loss / (i + 1)
