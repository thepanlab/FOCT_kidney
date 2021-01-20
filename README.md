# README

Code for the soon to be submitted paper: "Forward Optical Coherence Tomography Endoscope for Percutaneous Nephrostomy Guidance" 
The following pieces of python code and jupyter notebooks. It will guide you to the use of:
* Resnet 34
* Resnet 50 and Mobilenetv2 with and without pretrained initial weights from Imagenet Dataset.

# Prerequisites

The language used is Python. We used Tensorflow 2.3.

# Structure:
* *0-Read_images.ipynb* <br>
    It process the images from JPEG to numpy ndarray binaries
* *ResNet34/* <br>
    * *archResNet_p1.py* <br>
    * *archResNet_p2.py* <br>
    * *archResNet_p3.py* <br>
    * *archResNet_p4.py* <br>

    It uses the ResNet34 architecture to predict the type of tissue( 3 categories)
    It is split in 4 files in order to be able to run them independently.

* *ResNet50/* <br>
    * *Resnet50_batch/* <br>
        * *resnet50_arg_simult.batch* <br>
    * *Resnet50_python/* <br>
        * *archResNet50_arg.py* <br>

    The python file is used as:
    *archResNet50_arg.py testing_kidney validation_kidney*
    e.g.
    > archResNet50_arg.py 1 2

    The batch file was used in Summit supercomputer.

# Contact

Paul Calle - pcallec@ou.edu <br>
Project link: https://github.com/thepanlab/FOCT_kidney