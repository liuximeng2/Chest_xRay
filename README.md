# Chest X-Ray Classification Machine Learning Project
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview-of-the-project">Overview of the Project</a></li>
    <li><a href="#description-of-the-raw-dataset">Description of the Raw Dataset</a></li>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#description-of-the-processed-dataset">Description of the Processed Dataset</a></li>
    <li><a href="#modeling">Modeling</a></li>
    <ul>
        <li><a href="#knn">KNN</a></li>
        <li><a href="#principal-component-regression">Principal Component Regression</a></li>
    <li><a href="#random-forest">Random Forest</a></li>
    <li><a href="#convolutional-neural-network">Convolutional Neural Network</a></li>
      </ul>
  </ol>
</details>

## Overview of the Project
The goal of this project is to build a machine learning model that can classify chest X-ray images into normal and pneumonia. The dataset is from Kaggle and can be found [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). 

The minor distinction between healthy and bacterial pneumonia chest X-rays presents considerable challenges for image classification, offering substantial scope for the application and evaluation of various machine learning models.

We consider the following models:
- K-Nearest Neighbors
- Principal Component Analysis with Logistic Regression
- Random Forest
- Convolutional Neural Network

## Description of the Raw Dataset

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Data Preprocessing
### Regarding the validation set  
Given that the provided validation set has relatively few images, we decided to prioritize the method of cross-validation in the training dataset over the validation set. 

```python
import sklearn.model_selection as skm
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=123)
```

### Regarding the training and test set  
1. The two folders have subfolders dividing them into NORMAL and PNEUMONIA images. However, for the purpose of training the model, we need to have a single folder or list containing all the images. To do so, we use the os and glob modules to read the images from the subfolders and save them into a single folder.

2. We need to replace the labels of the images with 0 and 1. To do so, we create a dictionary with the key being the label and the value being the corresponding number. Then, we use the map function to replace the labels with the numbers.

```python
code = {'NORMAL':0 ,'PNEUMONIA':1}
```

3. The images are large in size, which will take a long time to train the model. Therefore, we need to resize the images to a smaller size. To do so, we use the cv2 module to read the images and resize them to 64x64 pixels. We then save the data into a numpy array. They can be seen on the github repository under Chest_xRay/Daata/data_transformed.

The following code is used to perform the above steps for the training and test set.  

```python
import cv2
import glob as gb
import os
import numpy as np
#the directory that contain the train images set
trainpath='/Users/conny/Desktop/QTM 347/FinalProject/mydata/train/'

X_train = []
y_train = []
for folder in  os.listdir(trainpath) : 
    #gb.glob returns a list of all paths matching a pathname, here it returns a list of all the images in the folder
    files = gb.glob(pathname= str( trainpath + folder + '/*.jpeg'))
    for file in files: 
        image = cv2.imread(file)
        #resize images to 64 x 64 pixels
        image_array = cv2.resize(image , (64,64))
        X_train.append(list(image_array))
        y_train.append(code[folder])
np.save('X_train',X_train)
np.save('y_train',y_train)
```

```python
#the directory that contain the train images set
testpath='/Users/conny/Desktop/QTM 347/FinalProject/mydata/test/'

X_test = []
y_test = []
for folder in  os.listdir(testpath) : 
    files = gb.glob(pathname= str( testpath + folder + '/*.jpeg'))
    for file in files: 
        image = cv2.imread(file)
        #resize images to 64 x 64 pixels
        image_array = cv2.resize(image , (64,64))
        X_test.append(list(image_array))
        y_test.append(code[folder])
np.save('X_test',X_test)
np.save('y_test',y_test)
```
</br>

Let us check the shape of the training set to better understand what the data looks like at this point.
```python
print('Train Data Set Shape = {}'.format(np.array(X_train).shape))
print('Train Labels Shape = {}'.format(np.array(y_train).shape))
```
![Training Shape](https://github.com/liuximeng2/Chest_xRay/blob/main/PCR/PCR_Images/train_shape.png)

This illustrates that we have successfully converted the images into a numpy array and the labels into a list of numbers. 5216 shows that we have 5216 images in the training set. 64, 64, 3 shows that the images are 64x64 pixels and have 3 channels. X-ray images are all in grayscale, so that we can convert the 3 channels into 1 channel to save space and improve computational efficiency.


## Description of the Processed Dataset
In the previous section, we talked about how the data is transformed and stored on github. Next we perform some data exploration to better understand the data.

- First, let us look at some of the images to know what we are dealing with. Is it possible to tell the difference between a normal and pneumonia patient by naked eyes?
```python
#plotting images of NORMAL and PNEUMONIA
plt.figure(figsize=(20,10))
for n , i in enumerate(np.random.randint(0,len(loaded_X_train),16)): 
    plt.subplot(2,8,n+1)
    plt.imshow(loaded_X_train[i])
    plt.axis('off')
    plt.title(getcode(loaded_y_train[i]))
```
![Chest Images](https://github.com/liuximeng2/Chest_xRay/blob/main/PCR/PCR_Images/Chest%20Images.png)

As we can see, it is quite difficult to tell the difference between the two types of images. This is why we need to use machine learning to help us classify the images.

- Next, let us look at the distribution of the labels in the training set.

```python
#count plot to show the number of pneumonia cases to normal cases in the train data set
df_train = pd.DataFrame()
df_train["labels"]= loaded_y_train
lab = df_train['labels']
dist = lab.value_counts()
dist = pd.DataFrame({'Label': dist.index, 'Count': dist.values})
plt.bar(dist['Label'], dist['Count'], color ='maroon', 
        width = 0.4, tick_label = dist['Label'])
plt.show()
```
![Distributuon](https://github.com/liuximeng2/Chest_xRay/blob/main/PCR/PCR_Images/count.png)

As we can see, there are more pneumonia cases than normal cases in the training set. This is a potential issue that we might need to deal with later. Let us keep this in mind.

- Next, let us look at the pixel values of the images after we reduced the size of the images. Particularly, we want to see the distribution of the pixel values of the images. We will plot the distribution of the pixel values of a random image in the training set.

```python
def plotHistogram(a):
    plt.figure(figsize=(12, 6))
    
    # Display the grayscale image
    plt.subplot(1, 2, 1)
    plt.imshow(a, cmap='gray')
    plt.title('Grayscale Image Display', fontsize=15)

    # Histogram for the grayscale image
    histo = plt.subplot(1, 2, 2)
    histo.set_title('Pixel Intensity Distribution', fontsize=15)
    histo.set_ylabel('Count', fontsize=12)
    histo.set_xlabel('Pixel Intensity', fontsize=12)
    n_bins = 30

    # Plot histogram
    plt.hist(a[:,:,0].flatten(), bins=n_bins, lw=0, color='black', alpha=0.7)
    plt.hist(a[:,:,1].flatten(), bins=n_bins, lw=0, color='black', alpha=0.7)
    plt.hist(a[:,:,2].flatten(), bins=n_bins, lw=0, color='black', alpha=0.7)

    plt.tight_layout()  # Adjust the layout
plotHistogram(loaded_X_train[np.random.randint(len(loaded_X_train))])
```
![Histogram](https://github.com/liuximeng2/Chest_xRay/blob/main/PCR/PCR_Images/pixel_image.png)

This histogram shows that the pixel values are distributed between 0 and 255. Some of the greatest counts are at 0. The grayscale image shows the intensity of the pixels. The darker the pixel, the lower the intensity. The lighter the pixel, the higher the intensity. At 0, the pixel is completely dark. At 255, the pixel is completely white. 


## Modeling
In this section, we will talk about the models that we used to classify the images. We will talk about the models that we used, the hyperparameters that we tuned, and the results that we got.

Before that, we need to perform a final step of data preprocessing. We need to flatten the images into a 2d array, so that we can feed the data into the models. To this end, this 2d array will have the size of (5216, 4096). 5216 is the number of images in the training set. 4096 is the number of pixels in each image. 64x64x1 = 4096.

```python
#flatten the images into a 2d array, for model training and testing
X_test = loaded_X_test.reshape(624, 64*64)
X_train = loaded_X_train.reshape(5216, 64*64)
```

We then scale the data using StandardScaler. This is to make sure that the data is centered around 0 and has a standard deviation of 1. This is to make sure that the data is not biased towards any particular feature.

```python
#Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
```
While the above steps are applied to all the models, the name of the variables might be slightly different as a result of personal choices when we assign the tasks to each member of the group. However, the steps are the same.



## KNN

## Principal Component Regression
Principal Component Regression (PCR) is a regression method that uses Principal Component Analysis (PCA) to reduce the number of predictor variables. It is a method that is used to deal with multicollinearity. It is also a method that is used to deal with the curse of dimensionality.

Given the case that we have 4096 features, we want to reduce the number of features to a more manageable number. One of the important hyperparameters of PCR is the number of components. We will tune this hyperparameter to find the best number of components.

Then we start our principal component analysis. We use the following library to perform PCA and logistic regression. 

```python
import warnings
from sklearn.exceptions import ConvergenceWarning
import sklearn.model_selection as skm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore", category=ConvergenceWarning)
```

The only hyperparameter that we tuned for PCA is the number of components. We tried different numbers of components and see which one gives us the lowest validation error. We used 5-fold cross validation to get the validation error. We then use the best number of components to train the model and get the test error.

```python
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=123)
```

For the choice of the number of components, we tried a range from 10 to 200 with a step size of 10. The reason behind this range of choice is that empirical evidence shows that the optimal number of components usually lies around the square root of the number of features. In our case, the number of pixels flattened is 4096. So we need to ensure that the range include something around 64. We also need to make sure that the range is not too large, so that the model does not take too long to run.

```python
num_components_PCA = np.arange(10, 200, 10)
```

We then loop through the range of number of components and get the validation error for each number of components. We then choose the number of components that gives us the lowest validation error. We then use this number of components to train the model and get the test error.

```python
best_num_components = None
best_val_accuracy = 0
pcr_val_accuracy = []

for n in num_components_PCA:
    # Create a pipeline with PCA and logistic regression
    pca = PCA(n_components=n)
    logistic_reg = LogisticRegression()
    pipe = Pipeline([('pca', pca), ('logistic', logistic_reg)])

    scores = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring='accuracy')

    mean_accuracy = np.mean(scores)
    pcr_val_accuracy.append(mean_accuracy)

    if mean_accuracy > best_val_accuracy:
        best_val_accuracy = mean_accuracy
        best_num_components = n

pcr_val_error = 1 - np.array(pcr_val_accuracy)
bst_pcr_val_error = 1 - best_val_accuracy

# After the loop, best_num_components will have the number of components with the highest mean accuracy or the lowest mean error
print(f"Best Number of Components: {best_num_components}, with an average test error of: {bst_pcr_val_error}")
```

Best Number of Components: 140, with an average test error of: 0.04064167979928224.

Here is a close look into the change of validation error as the number of components increases.
![Trend](https://github.com/liuximeng2/Chest_xRay/blob/main//PCR/PCR_Images/PCR_trend.png)

```python
# Plot the validation error as a function of the number of components
plt.figure(figsize=(10,6))
plt.plot(np.arange(10, 200, 10), pcr_val_error, marker='o', linestyle='--', color='r')
plt.xlabel('Number of Components')
plt.ylabel('Validation Error')
plt.title('Validation Error vs Number of Components')
plt.show()
```
Let us retrain the model using the best number of components and get the test error.

```python
#Use the best number of components to fit the PCA model
pca = PCA(n_components=best_num_components)
pipe = Pipeline([('pca', pca), ('logistic', logistic_reg)])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
print(f"Test Error of PCR using the best number of components(140): {1 - pipe.score(X_test, y_test)}")
```
Test Error of PCR using the best number of components(140): 0.21153846153846156

-We can also look at the confusion matrix to see which classes are misclassified.

```python
#function to plot the confusion matrix for each model
def plot_cm(predictions, y_test, title):
  labels = ['Normal', 'Pnuemonia']
  labels_predicted = ['Predicted Normal', 'Predicted Pnuemonia']
  labels_actual = ['Actual Normal', 'Actual Pnuemonia']
  cm = confusion_matrix(y_test,predictions)
  cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
  plt.figure(figsize = (8,8))
  plt.title(title)
  sns.heatmap(cm, linecolor = 'black' , linewidth = 2 , annot = True, fmt='', xticklabels = labels_predicted, yticklabels = labels_actual)
  plt.show()

pipe_pred_pcr = pipe.predict(X_test)
plot_cm(pipe_pred_pcr, y_test, 'PCR Confusion Matrix')
```
![ConfusionMatrix](https://github.com/liuximeng2/Chest_xRay/blob/main/PCR/PCR_Images/PCR_cm.png)

- SMOTE(Synthetic Minority Oversampling Technique)
We can see that there are more false negatives than false positives. This is largely attributed to the fact that the dataset is imbalanced. There are more images of pneumonia than normal. So the model is more likely to predict an image as pneumonia. To resolve this, we can use SMOTE to oversample the minority class.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

In this way, the number of images of normal and pneumonia are the same. We can then retrain the model and get the test error.

```python
warnings.filterwarnings("ignore", category=ConvergenceWarning)

num_components_PCA = np.arange(10, 200, 10)
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=123)

best_num_components = None
best_val_accuracy = 0
pcr_val_accuracy = []

for n in num_components_PCA:
    # Create a pipeline with PCA and logistic regression
    pca = PCA(n_components=n)
    logistic_reg = LogisticRegression()
    pipe = Pipeline([('pca', pca), ('logistic', logistic_reg)])

    scores = cross_val_score(pipe, X_train_smote, y_train_smote, cv=kfold, scoring='accuracy')

    mean_accuracy = np.mean(scores)
    pcr_val_accuracy.append(mean_accuracy)

    if mean_accuracy > best_val_accuracy:
        best_val_accuracy = mean_accuracy
        best_num_components = n

# After the loop, best_num_components will have the number of components with the highest mean accuracy
print(f"Best Number of Components: {best_num_components}, with an average error of: {1 - best_val_accuracy}")
```
Best Number of Components: 190, with an average error of: 0.03251612903225798

```python
#Use the best number of components to fit the PCA model
pca = PCA(n_components=best_num_components)
pipe = Pipeline([('pca', pca), ('logistic', logistic_reg)])
pipe.fit(X_train_smote, y_train_smote)
pipe.score(X_test, y_test)
print(f"Test Error of PCR using the best number of components(190): {1 - pipe.score(X_test, y_test)}")
```
Test Error of PCR using the best number of components(190): 0.19551282051282048
From here, we see that the test error has decreased from 0.21 to 0.19. We can also look at the confusion matrix to see which classes are misclassified.

```python
pipe_pred_pcr_smote = pipe.predict(X_test)
plot_cm(pipe_pred_pcr_smote, y_test, 'PCR Confusion Matrix')
```

![ConfusionMatrix2](https://github.com/liuximeng2/Chest_xRay/blob/main/PCR/PCR_Images/smote_pcr_cm.png)
As we can see, the number of false negatives has decreased, while the number of false positives slightly increased. The overall test error has decreased.



## Random Forest

## Convolutional Neural Network


## Authors

Contributors names and contact info

Name: Junyi (Conny) Zhou  
Contact: junyi.zhou@emory.edu

Name: Simon Liu  
Contact:

Name: Zhengyi Ou  
Contact:

Name: David Cheng  
Contact:





## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
