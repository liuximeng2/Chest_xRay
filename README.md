# Chest X-Ray Classification Machine Learning Project

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#description of the raw dataset">Description of the Raw Dataset</a></li>
    <li><a href="#data preprocessing">Data Preprocessing</a></li>
    <li><a href="#description of the processed dataset">Description of the Processed Dataset</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Description of the Raw Dataset

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Data Preprocessing
- Regarding the validation set
Given that the provided validation set has relatively few images, we decided to prioritize the method of cross-validation in the training dataset over the validation set. 
For cross-validation, we use the KFold function from sklearn.model_selection to split the training dataset into 5 folds. To ensure replicability, we set the random_state to 123 across all models.

```python
import sklearn.model_selection as skm
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=123)
```

- Regarding the training and test set
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


Let us check the shape of the training set to better understand what the data looks like at this point.
```python
print('Train Data Set Shape = {}'.format(np.array(X_train).shape))
print('Train Labels Shape = {}'.format(np.array(y_train).shape))
```
![Training Shape](https://github.com/liuximeng2/Chest_xRay/blob/main/Images/train_shape.png)

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
![Chest Images](https://github.com/liuximeng2/Chest_xRay/blob/main/Images/Chest%20Images.png)

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
![Distributuon](https://github.com/liuximeng2/Chest_xRay/blob/main/Images/count.png)

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
![Histogram](https://github.com/liuximeng2/Chest_xRay/blob/main/Images/pixel_image.png)

This histogram shows that the pixel values are distributed between 0 and 255. Some of the greatest counts are at 0. The grayscale image shows the intensity of the pixels. The darker the pixel, the lower the intensity. The lighter the pixel, the higher the intensity. At 0, the pixel is completely dark. At 255, the pixel is completely white. 


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



## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
