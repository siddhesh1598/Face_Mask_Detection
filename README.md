# Face_Mask_Detection

Detecting faces from the webcam and to determine whether the person is wearing a mask or not. The code uses MobileNetV2 as a base model for transfer learning and is trained on Medical-mask-dataset from kaggle consisting of 678 images of people wearing/not wearing masks. 

## Technical Concepts
**MobileNetV2:** The paper can be found [here](https://arxiv.org/abs/1801.04381)


## Getting Started

Clone the project repository to your local machine, then follow up with the steps as required.

### Requirements

After cloning the repository, install the necessary requirements for the project.
```
pip install -r requirements.txt
```

### Training

The maskNet.model file is pre-trained in the images from the [Medical-Mask-Dataset](https://www.kaggle.com/vtech6/medical-masks-dataset). If you wish to train the model from scratch on your own dataset, prepare your dataset in the following way:
1. Load the images in the "*images*" folder
2. Load the xml files containing the bounding box co-ordinates and the labels "bad" or "good" in the "*labels*" folder (please check the xml files from the original dataset to get an idea of the iags used in the files)
3. Store the "*images*" and "*labels*" folder under the "*dataset*" folder

You can then train the model by using the train.py file
```
python train.py --dataset dataset
```
![alt text](https://github.com/siddhesh1598/Face_Mask_Detection/blob/master/plot.png?raw=true)

The plot for Training and Validation Loss and Accuracy.

### Testing

To test the model on your webcam, use the main.py file. 
```
python main.py
```

You can pass some optional parameters for the main.py file:
1. --face: path to face detector model directory <br>
          *default: face_detector*
2. --model: path to trained face mask detector model <br>
          *default: maskNet.model*
3. --confidence: minimum probability to filter weak detections <br>
          *default: 0.35*



## Authors

* **Siddhesh Shinde** - *Initial work* - [SiddheshShinde](https://github.com/siddhesh1598)


## Acknowledgments

* Dataset: [Medical-Mask-Dataset](https://www.kaggle.com/vtech6/medical-masks-dataset) <br>
Dataset by **Eden Social Welfare Foundation, Taiwan**. (Re-uploaded by [Mikolaj Witkowski](https://www.kaggle.com/vtech6))
