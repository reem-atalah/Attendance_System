# Attendance_System
An image processing project uses Viola-jones technique to detect faces and then use LPB algorithm for recognition.

# Face Detection Using Viola-Jones Algorithm 
### steps to implement:
<ol>
  <li>Calculating Integral Image</li>
  <p>integral images are used to simplify the calculations of haar like features and save time of iterating over all pixels</p>
  <img align="center" src="https://miro.medium.com/max/1400/1*o4Bqfss5VoACUFKOvexnLA.jpeg" width=400/>
  </br>
  <li>Calculating Haar like features</li>
  <p>there are multiple types of haar feature windows, in our implementation, we used 5 types:
    (2 horizontal, 2 vertical, 3 horizontal, 3 vertical, 2*2 diagonal)</p>
  
  <img align="center" src="https://miro.medium.com/max/875/1*QOLDt87T8DT-6bJpVV_raA.png" width= 400/>
  </br>
  <li>AdaBoost Learning Algorithm</li>
  <img align="center" src=https://miro.medium.com/max/1200/1*4CGCobq9JWvZZMHL7D7fCg.png width= 400/>
  <li>Cascade Filter</li>
  <img align="center" src=https://miro.medium.com/max/1400/1*an9QzEqvqY9PVssknJOCrw.png width= 400/>
</ol>

# Preprocessing Images
### before preprocessing
![output4](https://user-images.githubusercontent.com/56982963/148285026-3c41c979-1bd5-4480-86b2-7708feab3c91.png)
#### - image is cropped centered and resized to fit our window size(19,19)
#### - a gamma correction is applied to the image to enhance the detection
![output3](https://user-images.githubusercontent.com/56982963/148284897-98b3ac48-6e8e-4ee3-b6b4-7330ecdbb902.png)

# Results 📝
### Best Accuracy we got from a model trained by a training set of ( 2000 faces, 1500 non-faces) with 40 classifiers and only 1 layer of cascade classifier
### ![test](https://user-images.githubusercontent.com/56982963/148281772-ec127377-3b03-49be-97a9-2adab95a743f.png)
### results on image with multiple faces 👥
![output2](https://user-images.githubusercontent.com/56982963/148283237-cddbe5d2-f341-4995-a295-65170411dd41.png)
### results on realtime with only one face 👤

![IMG-20220105-WA0024](https://user-images.githubusercontent.com/56982963/148284081-83df6223-1f01-4564-92be-3115e274ed6e.jpg)

# How to use?
#### open ViolaJones/main.ipynb and run all cells



# Recognition 📝
### can detect most of input images
### results on image  👥
![output2](https://github.com/reem-atalah/Attendance_System/blob/main/report/BenzemaDetect.png?raw=true)

# How to use?
#### make new dataset in gray images and name it with your name
#### save your input photo in images folder
#### open recognize faces.ipynb and run all cells and just type your photo name and extention in test function 
### For Training
<ol><li> In the fisrt cell slice the dataset as you want (the pkl file consist of 4000 faces and 7060 non-faces)</li>
<li>Run the second cell and wait until the model finish the training (it might take a while depending on number of training samples)</li>
<li>after that the model is stored in file called cvj_weights-..-...-...pkl  </li>
  </ol>
  
### For Testing 
<ol><li> Run the remaining cells and change the image by the one you want </li>
<li> For realtime run the last cell that opens camera for you</li>
  </ol>
