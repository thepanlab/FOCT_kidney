# README

Code for the paper: "Deep-learning-aided forward optical coherence tomography endoscope for percutaneous nephrostomy guidance"[1] 
The following pieces of python code and jupyter notebooks were used for the paper. The following architectures were used:
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

* *Processing_results.ipynb* <br>
    Processing of the results to obtain the accuracies, epochs of all the combinations. Time is calculated for a few combinations

* *Processing_predictions.ipynb* <br>
    Processing of the predictions to obtain the ROC curves

* *Processing_time.ipynb* <br>
    Complete processinf of time for cross-validation.

* *Grad-CAM.ipynb* <br>
    Implementation of visual explanation using Grad-CAM[2] for the models obtained

For *ResNet34* run the python code, for the rest you need to use arguments.    The python file is used as: <br>
> archResNet50_arg.py testing_kidney validation_kidney
    
e.g.

> archResNet50_arg.py 1 2

The batch file was used in Summit supercomputer.

# Paper
[1] Chen Wang, Paul Calle, Nu Bao Tran Ton, Zuyuan Zhang, Feng Yan, Anthony M. Donaldson, Nathan A. Bradley, Zhongxin Yu, Kar-ming Fung, Chongle Pan, and Qinggong Tang, "Deep-learning-aided forward optical coherence tomography endoscope for percutaneous nephrostomy guidance," Biomed. Opt. Express 12, 2404-2418 (2021) 

[Paper link](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-12-4-2404&id=449681)

# References
[2] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

# Contact

Paul Calle - pcallec@ou.edu <br>
Project link: https://github.com/thepanlab/FOCT_kidney
