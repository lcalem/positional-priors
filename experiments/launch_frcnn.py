import argparse

import os
import sys

from config import config_utils

# Import Mask RCNN
from model.mrcnn.config import Config
from model.mrcnn import utils as mutils
import model.mrcnn.model as modellib

from experiments import launch_utils as utils

from data.openimage.oid_first import OIDataset
from data.pascalvoc.pascal_frcnn import PascalVOCDataset


# Directory to save logs and trained model
MODEL_DIR = os.path.join(os.environ['HOME'], 'prior_experiments')

# Local path to trained weights file (TODO this is shit)
COCO_MODEL_PATH = os.path.join(os.environ['HOME'], 'positional-priors', 'experiments', 'frcnn', 'mask_rcnn_coco.h5')
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print('downloading coco weights')
    mutils.download_trained_weights(COCO_MODEL_PATH)


from config.config import cfg
from pprint import pprint


ALL_PCT = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


class Launcher():

    def __init__(self, exp_folder, percent=100, init_weights=None):
        '''
        exp_percents: the known label percentages of the sequential experiments to launch (default: 100)
        '''

        self.exp_folder = exp_folder   # still not sure this should go in config or not
        self.data_dir = cfg.DATASET.PATH
        self.relabel = False
        self.init_weights = init_weights
        assert (init_weights in ['coco', 'imagenet']) or (os.path.isfile(init_weights) and init_weights.endswith('.h5'))

        if percent is None:
            self.exp_percents = ALL_PCT
        elif isinstance(percent, int):
            assert percent in ALL_PCT
            self.exp_percents = [percent]
        elif isinstance(percent, str):
            parts = [int(p) for p in percent.split(',')]
            assert all([p in ALL_PCT for p in parts])
            self.exp_percents = parts

        print('Launching with percentages %s' % str(self.exp_percents))

    def launch(self):
        '''
        launch one experiment per known label proportion
        '''
        for p in self.exp_percents:
            print('\n=====================\nLaunching experiment for percentage %s \n' % p)

            # made two separate functions to avoid clutter
            if self.relabel:
                self.launch_percentage_relabel(p)
            else:
                self.launch_percentage(p)
            print('\n=====================')

    def launch_percentage(self, p):
        '''
        For a given known label percentage p:

        1. load dataset
        3. callbacks
        4. load / build model
        5. train
        '''

        self.dataset_train = self.load_dataset(mode=cfg.DATASET.TRAIN, batch_size=cfg.BATCH_SIZE, p=p)
        # self.dataset_train = self.load_dataset(mode=self.config.DATASET_TEST, batch_size=self.config.BATCH_SIZE)
        self.dataset_test = self.load_dataset(mode=cfg.DATASET.TEST, batch_size=cfg.TEST_BATCH_SIZE)
        # self.dataset_train = self.load_dataset(mode=cfg.DATASET.TRAIN, batch_size=cfg.BATCH_SIZE, p=p)
        # self.dataset_test = self.load_dataset(mode=cfg.DATASET.TEST, batch_size=cfg.TEST_BATCH_SIZE)

        # callbacks
        # cb_list = self.build_callbacks(p)

        # model
        self.build_model(self.dataset_train.nb_classes, p)

        # Fine tune all layers
        self.model.train(self.dataset_train,
                         self.dataset_test,
                         learning_rate=cfg.TRAINING.START_LR,
                         epochs=cfg.TRAINING.NB_EPOCH,
                         layers="all")

        # # cleaning (to release memory before next launch)
        # K.clear_session()
        # del self.model

    def load_dataset(self, mode, batch_size, p=None):
        '''
        TODO: when regular cfg is used fallback on the switch
        '''
        if cfg.DATASET.NAME == 'oid':
            dataset = OIDataset()
            dataset.load_oid(self.data_dir, batch_size, mode, cfg=cfg)
            dataset.prepare()

            dataset = PascalVOC(self.data_dir, batch_size, mode, x_keys=['image', 'image_id'], y_keys=['multilabel'], p=p)
        elif cfg.DATASET.NAME == 'pascal':
            dataset = PascalVOCDataset()
            dataset.load_pascal(self.data_dir, batch_size, mode, cfg=cfg)
            dataset.prepare()

        elif cfg.DATASET.NAME == 'pascal_extended':
            dataset = PascalVOCDataset()
            dataset.load_pascal(self.data_dir, batch_size, mode, extended=True, cfg=cfg)
            dataset.prepare()

        else:
            raise Exception('Unknown dataset %s' % cfg.DATASET.NAME)

        return dataset

    def build_model(self, n_classes, p):
        '''
        TODO uniformiser avec le vrai launch quand model est mergé
        '''
        print("building model")
        # Create model in training mode
        self.model = modellib.MaskRCNN(mode="training",
                                       cfg=cfg,
                                       exp_folder=self.exp_folder)

        # Starting weights
        if self.init_weights == "imagenet":
            self.model.load_weights(self.model.get_imagenet_weights(), by_name=True)
        elif self.init_weights == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            self.model.load_weights(COCO_MODEL_PATH,
                                    by_name=True,
                                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        else:
            # Load the last model you trained and continue training
            self.model.load_weights(self.init_weights, by_name=True)

    def build_callbacks(self, prop, relabel_step=None):
        '''
        prop = proportion of known labels of current run

        TensorBoard
        MAPCallback
        SaveModel
        LearningRateScheduler
        '''
        # log.printcn(log.OKBLUE, 'Building callbacks')
        # cb_list = list()

        # # # tensorboard
        # # logs_folder = os.environ['HOME'] + '/partial_experiments/tensorboard/' + self.exp_folder.split('/')[-1] + '/prop%s' % prop
        # # log.printcn(log.OKBLUE, 'Tensorboard log folder %s' % logs_folder)
        # # tensorboard = TensorBoard(log_dir=os.path.join(logs_folder, 'tensorboard'))
        # # cb_list.append(tensorboard)

        # # Validation callback
        # if cfg.CALLBACK.VAL_CB is not None:
        #     cb_list.append(self.build_val_cb(cfg.CALLBACK.VAL_CB, p=prop, relabel_step=relabel_step))
        # else:
        #     log.printcn(log.WARNING, 'Skipping validation callback')

        # # Save Model
        # cb_list.append(SaveModel(self.exp_folder, prop, relabel_step=relabel_step))

        # # Learning rate scheduler
        # cb_list.append(LearningRateScheduler(lr_scheduler))

        # return cb_list
        return list()


# python3 launch_frcnn.py -g 0 -o frcnn_oid_baseline
# python3 launch_frcnn.py -g 2 -o frcnn_pascal_extended -w imagenet
# python3 launch_frcnn.py -g 0 -o oid_baseline -w /home/caleml/partial_experiments/exp_20200115_1952_frcnn_baseline_oid_baseline/openimages20200115T2023/mask_rcnn_openimages_0200.h5
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', '-o', required=True, help='options yaml file')
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')
    parser.add_argument('--percent', '-p', default=100, help='the specific percentage of known labels')
    parser.add_argument('--exp_name', '-n', help='optional experiment name')
    parser.add_argument('--init_weights', '-w', default='coco', help='init weights path (coco, imagenet or some h5 path')

    # options management
    args = parser.parse_args()
    options = config_utils.parse_options_file(args.options)
    config_utils.update_config(options)
    config_utils.sanitize_config()

    print('\n========================')
    print('Final config\n')
    pprint(cfg)
    print('========================\n')

    # init
    exp_folder = utils.exp_init(' '.join(sys.argv), exp_name=(args.exp_name or args.options))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        launcher = Launcher(exp_folder, percent=args.percent, init_weights=args.init_weights)
        launcher.launch()
    finally:
        # cleanup if needed (test folders)
        pass   # TODO when cfg is back
        # if cfg.CLEANUP is True:
        #     log.printcn(log.OKBLUE, 'Cleaning folder %s' % (exp_folder))
        #     shutil.rmtree(exp_folder)
