# Artist Classification Neural Network

## Computational Data Science Project

### Objective

Develop a Convulutional Neural Network (CNN) that is able to differentiate between artist styles.

### Requirements

Python 3.11.x with the dependencies in `requirements.txt` installed.

### Dataset

We choose a dataset from artists that we like (mery, torino, yukien) that have a large sample of artwork. They have been scraped from image hosting sites and preprocessed such that all images are at most 400 pixels wide. (see: `scrape.ipynb`)

We perform feature extraction on the scraped data in `feature_extraction.ipynb`: edge_density, laplacian_variance, shannon_entropy, hs_colorfulness, color_spread, color_entropy, temp_mean, temp_stddev, gray_mean, gray_stddev, lbp_0, lbp_1, lbp_2, lbp_3, lbp_4, lbp_5, lbp_6, lbp_7, lbp_8, lbp_9. These are then compiled into `dataset.csv` from which we train our models.

These features not only may aid the accuracy of the CNN, but allow us to better understand the qualities of each artist. (see: `data_visualization.ipynb`)

### Models 

- `naive_model.ipynb`: Implements a neural network with only linear layers, with the image is input.
- `metrics_only_model`:  Implements a neural network with only linear layers, with the extracted features as input.
- `cnn_basic.ipynb`: Implements a basic CNN.
- `cnn_hyperparam_tuning.ipynb`: Performs hyperparameter tuning on the CNN.
- `cnn_dataaugment.ipynb`: Implements a CNN with data augmentation.
- `cnn_resnet.ipynb`: Implements a CNN that finetunes the Resnet 50 pretrained model
- `cnn_vgg16.ipynb`: Implements a CNN that finetunes the VGG16 pretrained model