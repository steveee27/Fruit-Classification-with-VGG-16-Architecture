# **Fruit Classification with VGG-16 Architecture**

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Architecture](#model-architecture)
  - [Model 1 - Baseline VGG-16](#model-1---baseline-vgg-16)
  - [Model 2 - Modified VGG-16](#model-2---modified-vgg-16)
  - [Model 3 - Optimized VGG-16](#model-3---optimized-vgg-16)
  - [Model 4 - Fully Enhanced VGG-16](#model-4---fully-enhanced-vgg-16)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results and Discussion](#results-and-discussion)
  - [Model Performance](#model-performance)
  - [Key Insights](#key-insights)
  - [Limitations](#limitations)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview
This project aims to classify images of fruits into four categories: **Acai**, **Acerola**, **Apple**, and **Avocado**. Using **Convolutional Neural Networks (CNN)**, the objective is to predict the fruit type from an image. The project utilizes **VGG-16** architecture as a baseline, and optimizes it with techniques like data augmentation, dropout, and batch normalization to improve the model's performance.

## Dataset Overview
The dataset for this classification task can be accessed [here](https://tinyurl.com/UTSDeepLearning2024No2). It contains **1600 images** divided into the following four fruit categories:
- **Acai**: 400 images
- **Acerola**: 400 images
- **Apple**: 400 images
- **Avocado**: 400 images

The images are resized to **224x224 pixels** and augmented to increase dataset diversity.

## Data Preprocessing
The following preprocessing steps were performed on the dataset:
- **Image Resizing**: All images were resized to **224x224 pixels**.
- **Data Augmentation**: Augmentation techniques like rotation, flipping, zooming, and shifting were applied.
- **Dataset Split**: The dataset was divided into:
  - **80%** for training
  - **10%** for validation
  - **10%** for testing

## Exploratory Data Analysis (EDA)
During the EDA process, the following was observed:
- **Image Histograms**: Distribution of color histograms across the images.
- **Image Quality**: Minor inconsistencies and noise were detected.
- **Data Balance**: The dataset is relatively balanced, though slight class imbalances were noted.

## Model Architecture
![image](https://github.com/user-attachments/assets/75741693-e942-4a94-92e6-77dadb2d09f4)

### Model 1 - Baseline VGG-16
The first model is based on **VGG-16**, a pre-trained CNN architecture. It includes 13 convolutional layers, followed by max-pooling layers, and fully connected layers with 4096 nodes.

- **Input Layer**: 224x224x3 images
- **Hidden Layers**: 13 convolutional layers with max-pooling
- **Output Layer**: Softmax for classification (4 categories)

Performance on the validation set: **25%** accuracy.

### Model 2 - Modified VGG-16
This model adds **batch normalization** and **dropout** to the baseline model to prevent overfitting and improve training.

- **Batch Normalization** was used to stabilize training.
- **Dropout** helps regularize the model.

Performance on the validation set: **46%** accuracy.

### Model 3 - Optimized VGG-16
Further improvements were made by increasing the number of filters to **1024** and implementing a **learning rate scheduler** for more efficient training. This optimized model also used **early stopping** to prevent overfitting.

Performance on the validation set: **74%** accuracy.

### Model 4 - Fully Enhanced VGG-16
In the final model, the **fully connected layers** were expanded to **8192 nodes** to increase the model's capacity. Additionally, **learning rate scheduling** and **early stopping** were used.

Performance on the validation set: **66%** accuracy.

## Model Training and Evaluation
The models were trained using **Adam optimizer** and **categorical cross-entropy** as the loss function. The **early stopping** technique was applied to halt training once performance started to plateau. Evaluation metrics included **accuracy**, **precision**, **recall**, and **F1-score**.

## Results and Discussion

### Model Performance
- **Model 1**: Baseline VGG-16 achieved an accuracy of **25%**.
- **Model 2**: Modified VGG-16 with dropout and batch normalization achieved **46%** accuracy.
- **Model 3**: Optimized VGG-16 with 1024 filters and learning rate scheduler achieved **74%** accuracy.
- **Model 4**: Fully Enhanced VGG-16 achieved **66%** accuracy.

The **Optimized VGG-16 (Model 3)** performed the best with the highest accuracy of **74%**, while **Model 4** was the second best with an accuracy of **66%**.

### Key Insights
1. **Data Augmentation** played a crucial role in improving generalization and preventing overfitting.
2. **Learning Rate Scheduling** and **Early Stopping** helped optimize the training process and improve model performance.
3. **Expanded Architecture** (in Model 4) did not always lead to better results, highlighting the importance of balancing model complexity.

### Limitations
- **Class Imbalance**: Despite the dataset being relatively balanced, smaller differences in class precision, especially for **Apple** and **Acerola**, suggest a need for further improvement.
- **Overfitting**: Even with early stopping, some signs of overfitting were present, particularly when expanding the model.

## Conclusion
This project demonstrates the application of **VGG-16** for classifying fruits into four categories. The **Optimized VGG-16** achieved the best results with **74%** accuracy. While the model performed well, future improvements could focus on increasing dataset diversity, adding more advanced architectures, and addressing class imbalances.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
