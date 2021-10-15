# Image Blur Detection
A blur detection model trained to detect motion and out of focus image blur implemented using pytorch lightning.

## Dataset

The model is trained on this [Blur Dataset](https://www.kaggle.com/kwentar/blur-dataset) from kaggle. The dataset consists of 1050 blurred and sharp images, consisting of 3x350 photos (motion-blurred, defocused-blurred, sharp). 

## CNN Architecture

The CNN model consists of 2x convolutional layers with pooling and dropout following 2x fully connected layers. For the first convolution layer a kernel size of 7x7 followed by a ReLU activation function, max pooling (2x2) and dropout (0.2) was chosen. In the second convolution layer a kernel of size 5x5 and again follow by ReLU, max pooling and dropout was applied. After the convolutions two fully connected layers (1024 units) are connected.

## Training the model

The model was designed with the help of pytorch lightning. For data preparation the training set was split into training, validation and test set with an 80/20 train/test and 90/10 validation/test split.
For training the EarlyStopping callback of pytorch lightning based on validation loss was used to avoid overfitting of the model.
A batch size of 128 was chosen and data augmentation in the train dataloader was performed.
As input a cropped image of size 96x96 pixels was chosen, depending on hardware constraints this could be increased.

## How to run

Install requirements:
```bash
$ pip install -r requirements.txt
```

Prepare datasets by splitting into train, validation and test set:
```bash
$ python prepare_labels.py <path_to_datafolder>
```

Train model:
```bash
$ python train.py
```
