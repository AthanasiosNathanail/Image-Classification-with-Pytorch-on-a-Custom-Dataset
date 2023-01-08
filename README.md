# Image-Classification-with-Pytorch-on-a-Custom-Dataset
A demonstration of a simple Image Classification model application on a custom dataset, for beginners or advanced Pythonistas.

This repository demonstrates the application of a custom image classification model based on Pytorch. This model was modified and adapted on a geological fossil dataset but it can be used for any image classification task on any custom dataset. 

I have only tested this model on a custom Anaconda environment on a Windows 10 PC.
The required packages for this tutorial can be installed by running the following command in your anaconda prompt.

## Installation

```bash
pip install -r requirements.txt
```

If you are a beginner with anaconda and environment management you can follow the instructions below.

To install requirements.txt in the environment, we have to use the pip installed within the environment. 
Thus, we should install pip first by

```bash
conda install pip
```

Then we can install the requirements.txt. Of course, there is a better way. 
We simply create a conda environment with pip installed in it:

```bash
conda create -n yourenv pip
```

Or if you want to specify your python version for this conda environment, conda create -n python=3.7 yourenv pip
Now you can run the following command to complete the installation.

```bash
pip install -r requirements.txt
```

For additional information please visit the link for the official anaconda documentation
1) https://www.anaconda.com/products/distribution (Anaconda installation)
2) https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html (Managing Anaconda Environments)
3) https://towardsdatascience.com/getting-started-guide-anaconda-80a4d30d3486 (Beginner's Anaconda Tutorial)

Once the anaconda environment is set up, you can download the code from this repository in a zip file, unzip it and place everything in folder of your choice

## Custom Dataset

To handle using custom datasets, torchvision provides a datasets.ImageFolder class.
ImageFolder expects data to be stored in the following way:
    
    root/class_x/x.png
    root/class_x/xx.png
    
    root/class_y/y.png
    root/class_y/yy.png

(works similarly for jpeg or jpg image format)

That is, each folder in the root directory is the name of a class, and within each of those folders are the images that correspond to that class. 
The images in the custom dataset used for this notebook are currently in the form of:
    
    repository main folder/Images/class1/1.jpg
    repository main folder/Images/class1/2.jpg   
    ...
    
    repository main folder/Images/class2/1.jpg
    repository main folder/Images/class2/2.jpg 
    ...

    repository main folder/Images/class3/1.jpg
    repository main folder/Images/class3/2.jpg 
    ...
    
Once you have assembled your dataset, we will need to split our data into train and test splits. To manually create a train, validation and test folder and store the relevant images in those folders we only need to create a train/val/test split once and re-use it each time we re-run the notebook. 

For this project I did a manual split on the data between train, validation and test data. The proportions of the dataset after the split are usually 70% training data, 20% validation data, and 10% test data. Of course, the choice of the split percentages is up to the user, but it is recommended to have at least 70% of the data for training.

After the successful split, there should be another folder in your projects directory with the following structure:

    # Train
    repository main folder/data/train/class1/1.jpg
    repository main folder/data/train/class1/2.jpg
    ...
    repository main folder/data/train/class2/1.jpg
    repository main folder/data/train/class2/2.jpg
    ...
    repository main folder/data/train/class3/1.jpg
    repository main folder/data/train/class3/2.jpg
    ...
    
    # Validation
    repository main folder/data/val/class1/1.jpg
    repository main folder/data/val/class1/2.jpg
    ...
    repository main folder/data/val/class2/1.jpg
    repository main folder/data/val/class2/2.jpg
    ...
    repository main folder/data/val/class3/1.jpg
    repository main folder/data/val/class3/2.jpg
    ...
    
    # Test
    repository main folder/data/test/class1/1.jpg
    repository main folder/data/test/class1/2.jpg
    ...
    repository main folder/data/test/class2/1.jpg
    repository main folder/data/test/class2/2.jpg
    ...
    repository main folder/data/test/class3/1.jpg
    repository main folder/data/test/class3/2.jpg
    ....
  

To further enhance the variability of the dataset used, certain common data augmentation techniques can be applied: randomly rotating, flipping horizontally and cropping the images.

For this project, a pretrained ResNet50 was used and the last layer of the model was trained with the user's dataset. That is because the pre-trained model is trained on a larger dataset with multiple classes, allowing the model to leanr numerous patterns and features. This type of learning is called Transfer Learning, allowing the model to perform more effectively compared to being trained from scratch.

The pre-trained weights are found in this repository but will also be automatically downloaded as we run the script. For more information on the weights and other backbone architectures, please see the official torchvision models page (https://pytorch.org/vision/stable/models.html). 

According to your hardware and computer capabilities, there are a number of parameters in the script that might need to be adjusted in order to successfully run the script of this project. The most important ones are the 'batch_size' and the 'number of workers'. 

I am using a RAM of 16GB, Processor: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz   3.79 GHz, and a GPU: NVIDIA GeForce RTX 2080 Super. 
The parameters used for the particular hardware were:
'batch_size': 16
'number of workers': 2

## Training the model

Preparing the model for training is set up in a simple way within the script. There are multiple parameters that the user can modify, but here is a list of the most important ones:

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())  # the learning rate can be selected along with the optimizer
max_epochs_stop=3,
n_epochs=100,
print_every=1

For a detailed explanation, you can refer to https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212.

Once the model is trained, a part of the script allows you to generate multiple figures to monitor the models training and losses. The weights of the trained model are then saved for future use and to run inference. 

  resnet50-transfer.pt
  resnet50-transfer.pth
  
## Inference 

On this last part we follow a series of steps to ultimately test our model's prediction accuracy on images from the custom dataset:

  1. Load the weights from the training step
  2. Use the trained model for Inference
  3. Display the model's predictions on a random test image
  4. Calculate the model's Test Accuracy
  5. Use a function to Evaluate Model Over All Classes
  6. Generate Confusion matrix
  7. Export results on a csv file
  8. Display results and models predictions on the desired images/directories
  
## Concluding Remarks
This project was a demonstration of how you can apply an Image Classification model from scratch on a custom dataset. For a detailed explanation of the entire project stay tuned for my post on Medium.

For a domain specific application and specific custom dataset please feel free to have a look on my other repository, in which I apply the model of this repository on geological data for Fossil classification.

If you have read this far, I really appreciate it. If you enjoyed this project and found it helpful, please share it so you can help another developer improve their projects.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## References

1. https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212
2. https://pytorch.org/vision/stable/models.html
3. https://www.anaconda.com/products/distribution
4. https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html 
5. https://towardsdatascience.com/getting-started-guide-anaconda-80a4d30d3486

## License

[MIT](https://choosealicense.com/licenses/mit/)
