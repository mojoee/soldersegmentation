class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'solder':
            return '../deeplabv3/DeepLab-V3/data/datasets/solder_spreading/solder_spreading_dataset_final/'  # folder that contains VOCdevkit/.

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
