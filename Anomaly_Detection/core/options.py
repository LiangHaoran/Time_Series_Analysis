"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
import argparse

class Options():
    """
    Options class
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='gammary', help='satellite | gens | shuttle | gammary ')
        self.parser.add_argument('--dataroot', default='/home/poac/code/Anomaly_Detection/signature_matrices/', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=128, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--imagC', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='(train, ad)')
        self.parser.add_argument('--model', type=str, default='framework', help='chooses which model to use. framework')
        self.parser.add_argument('--outf', default='/home/poac/code/Anomaly_Detection/experiments/', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')
        self.parser.add_argument('--hidden_size', type=int, default=5, help='hidden size of autoencoder')
        self.parser.add_argument('--num_layers', type=int, default=4, help='number layers of lstm autoencoder')
        self.parser.add_argument('--encode_window', type=int, default=100, help='the length of sliding window in encode')
        self.parser.add_argument('--test_file_path', type=str, default='/home/poac/code/Anomaly_Detection/root_cause_analysis/test_file.json', help='the path of label file')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--train_f_anogan', type=str, default='wgan')
        self.parser.add_argument('--noise', type=bool, default=False, help='add noise to test data')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--load_models', action='store_true', help='Load the pretrained models')
        self.parser.add_argument('--phase', type=str, default='train', help='train, ad, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr_d', type=float, default=1e-4, help='initial learning rate for netD')
        self.parser.add_argument('--lr_g', type=float, default=4e-4, help='initial learning rate for netG')
        # options for GAN-defect
        self.parser.add_argument('--image_size', type=int, default=128)
        self.parser.add_argument('--steps', type=list, default=[100, 200])
        self.parser.add_argument('--lrs', type=float, default=1e-2)
        self.parser.add_argument('--nBottleneck', type=int, default=4000)
        self.parser.add_argument('--defect_mode', type=str, default='geometry')
        self.parser.add_argument('--contrast_loss_weight', type=int, default=1)
        self.parser.add_argument('--d_every', type=int, default=1)
        self.parser.add_argument('--g_every', type=int, default=1)
        self.parser.add_argument('--s_every', type=int, default=5)
        self.parser.add_argument('--s_start', type=int, default=0)
        self.parser.add_argument('--decay_every', type=int, default=10)
        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        print('expriment dir:', expr_dir)
        ad_dir = os.path.join(self.opt.outf, self.opt.name, 'ad')
        print('ad dir:', ad_dir)

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(ad_dir):
            os.makedirs(ad_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
