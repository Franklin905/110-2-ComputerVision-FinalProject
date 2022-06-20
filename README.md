
----------------------------------------------------------------
| 2022 Spring Computer Vision Final Project                    |
| Light-Weight Facial Landmark Prediction                      |
| Team Name: VLL                                               |
| Team Members: Chen, Yu-Hsuan                                 |
|               Ji, Yan-Yang                                   |
|               Lai, Yung-Hsuan                                |
| Date: 2022/06/17                                             |
----------------------------------------------------------------

# Preparation
Follow the steps below to set up environment
1. Run the command: unzip R10942088.zip
2. Run the command: cd R10942088
3. Place data.zip under R10942088/ and run the command: unzip data.zip
4. Run the command: cd data
5. Place aflw_test.zip under data/, which is generated from the above step, and run the command: unzip aflw_test.zip
6. Run the command: cd ../
7. Set up an environment of python 3.8 (strongly recommended) and run the command: pip3 install -r requirements.txt
    - 7.1 If python version isn't 3.8, you might encounter error when loading .pkl file with pickle package. We guess TAs probably compressed the data in python 3.8 environment, so we recommend installing python 3.8. We encounter the error when running in python 3.6 environment.
    - 7.2 Possible solution: run the command: pip3 install pickle5. Modify the second line, "import pickle", in data.py to "import pickle5 as pickle"


# Training
Run the command below.
python3 DML_train.py --data_dir ./data --train_batch 64 --n_epoch 150 --lr_min 0.0025 --save_model_name rConvNext.pth


# Evaluation
12255average_model_best.pth is our best model trained by DML and weight avg.
Run the command below. Predicted facial landmarks of testing data will be written in solution.txt.
python3 test.py --data_dir ./data --train_batch 128 --save_model_name 12255average_model_best.pth


# Visualization
Run the command below. 50 visualized pictures for training data, validation data, testing data each will be saved in vis_folder.
python3 visualization.py --data_dir ./data --save_dir vis_folder --save_model_name 12255average_model_best.pth --num_pic 50


# Our Environment
 - OS:  Ubuntu 20.04.2 LTS
 - CPU: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
 - GPU: Nvidia TITAN RTX 24GB


# Contact
If you have any problems reproducing our work, please contact us by e-mail.
 - Chen, Yu-Hsuan : r10942088@ntu.edu.tw
 - Ji, Yan-Yang   : r10942090@ntu.edu.tw
 - Lai, Yung-Hsuan: r10942097@ntu.edu.tw