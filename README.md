# Font Classification

> This repository contains code for classifying fonts using CNN techniques. Font classification involves identifying the font family or typeface of a given text image. This can be useful for various applications such as font recognition in images, document analysis, and typography-related projects. A few sample images

<img width="1348" alt="image" src="https://github.com/mishra-kunal1/font_classification/assets/99056351/a06dc514-988d-4c86-803b-da8ab9739c00">

## Dataset
The dataset used for font classification consists of a collection of text images in corresponding folders indicating the font family or typeface. Each image in the dataset represents a sample of text written in a specific font.<br>
**There are total 10 font classes present in the dataset with approximate 80 sample images per font.**
The data is present at [project_files/data](https://github.com/mishra-kunal1/font_classification/tree/main/project_files/data) of the repo.
The folder structure looks like this 

<div align="left">
  <img width="180" alt="image" src="https://github.com/mishra-kunal1/font_classification/assets/99056351/cd7f7555-1726-4a78-aa7d-6b0837a77421">
</div>


## Generating more data

The dataset also contains .ttf file for each font which can be used to generate synthetic images.<br>

> The process of generating synthetic data is explanied in [generating_synthetic_images.ipynb](https://github.com/mishra-kunal1/font_classification/blob/main/notebooks/generating_syntetic_images.ipynb)

After creating synthetic dataset we have a total of 10,000 training images 1000 validation images, an ample amount suitable for training CNN models effectively.

## Models 

## Performance Metrics

## Training and Results

## Installation and Usage

### 1. Clone the GitHub repo:<br />
   ```
    git clone https://github.com/mishra-kunal1/font_classification.git
   ```
   
### 2. Change the dir and Install the requirements:<br />
   ```
   cd font_classification
   pip install -r requirements.txt
   ```

### 3. Prepare the dataset: <br />
   ```
   python prepare_dataset.py
   ```

### 4. Training the model: <br />
   #### 4.1 To start training the data from scratch using LeNet:<br />
   ```
   python train.py --model lenet
   ```

   #### 4.2 To Fine tune the model using ResNet18:<br />
   *Make sure to remove the grayscale transformation in the train.py file.*<br />
   ```
   python train.py --model resnet
   ```
   
   #### 4.3 To Fine tune the model using EfficientNet B1 :<br />
  *Make sure to remove the grayscale transformation in the train.py file.* <br />
  ```
   python train.py --model enet
   ```

   #### 4.4 To resume training from last checkpoint <br />
  ```
   python train.py --model resnet --resume yes
   ```

### 5. Evaluating the performance of test data <br />
  **Important  - Make sure to add the test_data path to inference.py before running the code.** <br />

  ```
    python inference.py
  ```
    








