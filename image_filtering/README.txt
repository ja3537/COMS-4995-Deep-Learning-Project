README

GenerateFlashImage.m and GenerateTrainingImages.m are Matlab files to generate the colored flash iamges.

get_metrics.ipynb takes the ground truth images and a folder of model output imaegs and computes PSNR, VGG loss, and SSIM.

train.py and train_attention_resunet.py can be called with python from the command line to train and save a model that takes in a noisy RGB image and a single guide image.

training and image generation for the models that take in no RGB image and three flash guide images are done in the other .ipynb files.

The rest of the python files are utilities to help with either dataset generation or metrics evaluation.