# Solder Segmentation
This repository is part of the publication regarding segmenting solder joints via image segmentation methods.

Take a look at the input image:

![Input Image](https://github.com/mojoee/soldersegmentation/blob/master/img.png "Input Image")

and now take a look at our resulting segmentation map:

![Sample Segmentation](https://github.com/mojoee/soldersegmentation/blob/master/result_image.png "Solder Segmentation")

# Dataset

The dataset can be accessed after sending the filled form solder_segmentation-dataset-license-agreement_Filed_28-03-2022_Contract.docx to soldersegmentation@gmail.com


Contact: Moritz Sontheimer

The Solder Segmentation (SolderSeg) Dataset contains 290 microscopic solder images.


Each image contains a different solder joint, and is given a unique name. Each image also comes with groundtruth labels, that are split into 3 regions, background, transition area and solder area.

This dataset can be used for image segmentation. For fair comparison, we randomly split the images into three sets, one for trainning, one for validation and one for testing.

Please cite our paper if you find this dataset useful for your research.

@article{,
  title={Deep Learning Based Morphological Solder Segmentation},
  author={Sontheimer, Moritz, Shou-Yuan Chou},
  journal={ICPRS2022},
  volume={},
  number={},
  pages={},
  year={2022},
  publisher={}
}

All Image rights belong to NTUST.

# GUI

We also share our GUI. How to use the GUI can be seen here: [https://youtu.be/cGnSzOiCfyI](https://youtu.be/cGnSz0iCfyI)

To use the GUI, you need to download some of our trained weights from here: https://drive.google.com/file/d/1pgx7AeYF3tjkKXZnKrZKhoD8J35A7-Qo/view?usp=sharing

To run the gui, run the img_viewer.py file
