from pynet.core import Base
from pynet.datasets.core import DataManager
from pynet.models.densenet import *
from pynet.models.fc import *
from pynet.losses import *
from pynet.augmentation import *
import pandas as pd
import re
from pynet.transforms import *
from configs.general_config import CONFIG

class BaseTrainer():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.num_classes, args, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args)
        self.loss = BaseTrainer.build_loss(args.loss, net=self.net, args=self.args)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=args.gamma_scheduler, step_size=args.step_size_scheduler)

        model_cls = Base

        self.model = model_cls(model=self.net,
                               metrics=args.metrics,
                               pretrained=args.pretrained_path,
                               freeze_until_layer=args.freeze_until_layer,
                               use_cuda=args.cuda,
                               loss=self.loss,
                               optimizer=self.optimizer)

    def run(self):
        with_validation = (self.args.nb_folds > 1)
        train_history, valid_history = self.model.training(self.manager,
                                                           nb_epochs=self.args.nb_epochs,
                                                           scheduler=self.scheduler,
                                                           with_validation=with_validation,
                                                           checkpointdir=self.args.checkpoint_dir,
                                                           exp_name=self.args.exp_name,
                                                           fold_index=self.args.folds,)

        return train_history, valid_history

    @staticmethod
    def build_loss(name, net=None, args=None):
        if name == 'l1':
            loss = nn.L1Loss()
        elif name == 'BCE':
            loss = nn.BCEWithLogitsLoss()
        else :
            loss = None
        return loss

    @staticmethod
    def build_network(name, num_classes, args, **kwargs):
        if name == "densenet121":
            net = densenet121(progress=False, num_classes=num_classes, drop_rate=args.dropout, bayesian=args.bayesian,
                              concrete_dropout=args.concrete_dropout, **kwargs)
        elif name in ["densenet121_block%i"%i for i in range(1,5)]+['densenet121_simCLR', 'densenet121_sup_simCLR']:
            block = re.search('densenet121_(\w+)', name)[1]
            net = densenet121(progress=False, num_classes=num_classes, drop_rate=args.dropout, bayesian=args.bayesian,
                              concrete_dropout=args.concrete_dropout, out_block=block, **kwargs)
        elif name == "fc":
            net = FCNet(num_classes=num_classes, drop_rate=args.dropout)
        else:
            raise ValueError('Unknown network %s' % name)
            ## Networks to come...
            # net = UNet(1, in_channels=1, depth=4, merge_mode=None, batchnorm=True,
            #            skip_connections=False, down_mode="maxpool", up_mode="upsample",
            #            mode="classif", input_size=(1, 128, 144, 128), freeze_encoder=True,
            #            nb_regressors=1)
            # net = SchizNet(1, [1, 128, 128, 128], batch_size)

        return net

    @staticmethod
    def get_data_augmentations(augmentations):
        if augmentations is None or len(augmentations) == 0:
            return None

        aug2tf = {
            'flip': (flip, dict()),
            'blur': (add_blur, {'snr': 1000}),
            'noise': (add_noise, {'snr': 1000}),
            'resized_crop': (Crop((115, 138, 115), "random", resize=True), dict()),
            'affine': (affine, {'rotation': 5, 'translation': 10, 'zoom': 0}),
            'ghosting': (add_ghosting, {'intensity': 1, 'axis': 0}),
            'motion': (add_motion, {'n_transforms': 3, 'rotation': 40, 'translation': 10}),
            'spike': (add_spike, {'n_spikes': 10, 'intensity': 1}),
            'biasfield': (add_biasfield, {'coefficients': 0.7}),
            #'swap': (add_swap, {'num_iterations': 20})

        }
        compose_transforms = Transformer()
        for aug in augmentations:
            compose_transforms.register(aug2tf[aug][0], probability=0.5, **aug2tf[aug][1])
        return [compose_transforms]


    @staticmethod
    def build_data_manager(args, **kwargs):
        labels = args.labels or []
        add_to_input = None
        data_augmentation = BaseTrainer.get_data_augmentations(args.da)
        self_supervision = None  # RandomPatchInversion(patch_size=15, data_threshold=0)
        input_transforms = kwargs.get('input_transforms')
        output_transforms = None
        patch_size = None
        input_size = None

        projection_labels = {
            'diagnosis': ['control', 'FEP', 'schizophrenia', 'bipolar disorder', 'psychotic bipolar disorder']
        }

        stratif = CONFIG['db'][args.db]


        ## Set the preprocessing step with an exception for GAN
        if input_transforms is None:
            if args.preproc == 'cat12': # Size [121 x 145 x 121], 1.5mm3
                if args.net == "alpha_wgan":
                    input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),
                                        HardNormalization()]
                else:
                    input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), Normalize()]
            elif args.preproc == 'quasi_raw': # Size [122 x 146 x 122], 1.5mm³
                if args.net == "alpha_wgan":
                    input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),
                                        HardNormalization()]
                else:
                    input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),
                                        Normalize()]

        ## Set the basic mapping between a label and an integer
        df = pd.concat([pd.read_csv(p, sep=',') for p in args.metadata_path], ignore_index=True, sort=False)

        # <label>: [LabelMapping(), IsCategorical]
        known_labels = {'age': [LabelMapping(), False],
                        'sex': [LabelMapping(), True],
                        'site': [
                            LabelMapping(**{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))}),
                            True],
                        'diagnosis': [LabelMapping(schizophrenia=1, FEP=1, control=0,
                                                   **{"bipolar disorder": 1, "psychotic bipolar disorder": 1}), True]
                        }

        assert set(labels) <= set(known_labels.keys()), \
            "Unknown label(s), chose from {}".format(set(known_labels.keys()))

        assert (args.stratify_label is None) or (args.stratify_label in set(known_labels.keys())), \
            "Unknown stratification label, chose from {}".format(set(known_labels.keys()))

        strat_label_transforms = [known_labels[args.stratify_label][0]] \
            if (args.stratify_label is not None and known_labels[args.stratify_label][0] is not None) else None
        categorical_strat_label = known_labels[args.stratify_label][1] if args.stratify_label is not None else None
        if len(labels) == 0:
            labels_transforms = None
        elif len(labels) == 1:
            labels_transforms = [known_labels[labels[0]][0]]
        else:
            labels_transforms = [lambda in_labels: [known_labels[labels[i]][0](l) for i, l in enumerate(in_labels)]]

        dataset_cls = None

        manager = DataManager(args.input_path, args.metadata_path,
                              batch_size=args.batch_size,
                              number_of_folds=args.nb_folds,
                              add_to_input=add_to_input,
                              add_input=args.add_input,
                              labels=labels or None,
                              sampler=args.sampler,
                              projection_labels=projection_labels,
                              custom_stratification=stratif,
                              categorical_strat_label=categorical_strat_label,
                              stratify_label=args.stratify_label,
                              N_train_max=args.N_train_max,
                              input_transforms=input_transforms,
                              stratify_label_transforms=strat_label_transforms,
                              labels_transforms=labels_transforms,
                              data_augmentation=data_augmentation,
                              self_supervision=self_supervision,
                              output_transforms=output_transforms,
                              patch_size=patch_size,
                              input_size=input_size,
                              pin_memory=args.pin_mem,
                              drop_last=args.drop_last,
                              dataset=dataset_cls,
                              device=('cuda' if args.cuda else 'cpu'),
                              num_workers=args.num_cpu_workers)

        return manager