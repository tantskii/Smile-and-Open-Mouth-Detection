Smile and Open Mouth Detection
============

* **Solutions**:
    * *solution 1*: end2end solution based on multitask MobileNetv2
    * *solution 2*: heuristic solution based on points of the mouth, its width, length and deviations from the line of the mouth
    * *solution 3*: solution based on multitask MLP above mouth, eye, eyebrow landmarks
    
* **Rare dependencies**:
    * [*jpeg4py*](https://github.com/ajkxyz/jpeg4py) - quick image reading
    * [*mtcnn*](https://github.com/ipazc/mtcnn) - face detection
    * [*imgaug*](https://imgaug.readthedocs.io/en/latest/index.html) - image augmentations

* For training it was used only [Multi-Task Facial Landmark (MTFL) dataset](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html) with own labels for open mouth (`AFLW.csv` and `net.csv`). It's necessary to place it in the `data` folder (`../data/MTFL/`)
* MULTI-PIE must be unpacked in the `data_test` folder
* Scripts for training and inference lie in the `scripts` folder, there you can specify your image folder for prediction.
* **MTFL test set f1 score**:
    * *solution 1*: smile - 0.87, open mouth - 0.93
    * *solution 2*: smile - 0.79, open mouth - 0.82
    * *solution 3*: smile - 0.82, open mouth - 0.83

* **Remarks**:
    * MULTI-PIE (300 images) f1 score is less, because it has a different image distribution and the labels isn't very good quality. Visually, the MobileNetv2 works well and fast.
    * MTFL images and smile labels also not very good quality which gives an underestimate
    * it's necessary to define more accurately when the mouth is open and when there is no
    * it's difficult for networks to define a smile; it's even more difficult for models which using face landmarks.
    * depending on the task, you need to choose a probability  threshold, in this case the emphasis was on precision

* **Not included**:
    * other face detectors - AAM, LBF, etc
    * own regression based neural network landmraks detector
    * pruning, quantization, tensorrt, etc
    * other approaches such as using mouth images only for speed, etc










