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
    * Cross-validation
        * *archResNet_p1.py* <br>
        * *archResNet_p2.py* <br>
        * *archResNet_p3.py* <br>
        * *archResNet_p4.py* <br>

    It uses the ResNet34 architecture to predict the type of tissue( 3 categories)
    It is split in 4 files in order to be able to run them independently.

* *PT_MobileNetv2/* <br>
    * Cross-validation
        * *PT_MobileNetv2_batch/* <br>
            * *mobilenetv2_tl_arg_simult_vC.batch*
        * *PT_MobileNetv2_python/* <br>
            * *mobilenetv2_tl_arg_vC.py*
* *ResNet50/* <br>
    * Cross-validation
        * *Resnet50_batch/* <br>
            * *resnet50_arg_simult.batch* <br>
        * *Resnet50_python/* <br>
            * *archResNet50_arg.py* <br>
    * Cross-testing
        * *Resnet50_batch/* <br>
            * *resnet50_arg_outer_simult.batch* <br>
        * *Resnet50_python/* <br>
            * *archResNet50_arg_outer.py* <br>
    
* *PT_ResNet50/* <br>
    * Cross-validation
        * *PT_Resnet50_batch/* <br>
            * *resnet50_tl_arg_simult.batch* <br>
        * *PT_Resnet50_python/* <br>
            * *archResNet50_tl_arg.py* <br>
    * Cross-testing
        * *PT_Resnet50_batch/* <br>
            * *resnet50_tl_arg_outer_simult.batch* <br>
        * *PT_Resnet50_python/* <br>
            * *archResNet50_tl_arg_outer.py* <br>


For *ResNet34* run the python code, for the rest you need to use arguments.    The python file is used as: <br>
> archResNet50_arg.py testing_kidney validation_kidney
    
e.g.

> archResNet50_arg.py 1 2

The batch file was used in Summit supercomputer.

# Contact

Paul Calle - pcallec@ou.edu <br>
Project link: https://github.com/thepanlab/FOCT_kidney