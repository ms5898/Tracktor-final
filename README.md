# Tracktor-final
**Mingfei Sun (ms5898)**

- [Dataset Prepar](#Dataset-Prepare)
- [Get the tracking result on CVAT videos](#Get-the-tracking-result-on-CVAT-videos)

## Dataset Prepare
#### You can just download from the google cloud:
* Just download [COSMOS_data_tracktor](url) from the shared google cloud, it has all the data you need for the experiment in Tracktor

#### If you can not download from the google cloud, you can also prepare it by yourself:

1. **Prepare a folder to put the dataset in called ``COSMOS_data_tracktor``, so your project folder should like that:**
```
ProjectTrafficIntersection.COSMOS.V2
|---Tracktor
|---COSMOS_data_tracktor
```

2. **Prepare the dataset for training the Faster R-CNN which use in the Tracktor to predict the next location of the target:**
* make a folder called ``detection_dataset`` in ``COSMOS_data_tracktor`` and download [maskrcnn_lablled-bbox-train-val.tar](https://drive.google.com/open?id=17FkzKVmcCypZNkO6PfKXbLUxji-Cj0_g) and [main-dataset-V4-DTL-20190908.tar](https://drive.google.com/open?id=1lPQr4pkyKYgJtpV99wRmHuXKqw-a9_3V) unzip them in ``COSMOS_data/COSMOS_data_tracktor``.
* After you download the two dataset above you will find that they are in different format, you should make them in the same format, so you should run that:
```
python dataset/data_prepare.py -- root (your main-dataset-V4-DTL-20190908 folder) 
                               --output_dir (your output folder for changed format)
```
Example: for me this step is 
```
python dataset/data_prepare.py -- root ../COSMOS_data_tracktor/detection_dataset/main-dataset-V4-DTL-20190908
                               --output_dir ../COSMOS_data_tracktor/detection_dataset/detection_dataset3
```
After that you will find there are three folder in your ``COSMOS_data_tracktor/detection_dataset`` the first two are downloaded from the internet and the last one is created by running the script ``dataset/data_prepare.py``

* You can use the jupyter notebook ``Notebook_experiment/PyTorch Dataset.ipynb`` to check the detection dataset result

3. **Dataset for Tracking:**

* Download the [CVAT](https://drive.google.com/open?id=1b40mIfUziefByIwRiWQk-H9PxhjfPBHy)** and put that in ``COSMOS_data_tracktor``

* change the original format to [MOT format](https://motchallenge.net) by runing the jupyter notebook ``Notebook_experiment/process tracking data and show.ipynb``

* After that you will find all videos are saved as MOT format in ``COSMOS_data_tracktor/Track_data_cosmos/v_and_p``, ``COSMOS_data_tracktor/Track_data_cosmos/only_v`` and ``COSMOS_data_tracktor/Track_data_cosmos/only_p`` you can delete the folder ``CVAT`` if you want.

* Yoy can use ``Notebook_experiment/sequence dataset for test.ipynb`` to check the tracking data format

4. **Mask R-CNN result:**

* Download [Mask R-CNN result](https://drive.google.com/drive/folders/1KRdJQcaO2EuUWOLF4AjZO0sXqMhP5vcb) and put it in ``COSMOS_data_tracktor``

* You can process the mask output result by running:
```
python dataset/data_prepare.py -- root (your MaskRCNN folder) 
                               --output_dir (your output folder for changed format)
```
Example: for me this step is 
```
python dataset/data_prepare.py -- root ../COSMOS_data_tracktor/MaskRCNN
                               --output_dir ../COSMOS_data_tracktor/mask_detect_result
```

* You can delete ``MaskRCNN`` if you want 

* You can check your mask R-CNN output by using ``Notebook_experiment/Process mask rcnn and show the result.ipynb``

5. **You can check the dataset for Tracking by using ``Notebook_experiment/sequence dataset for test.ipynb``**

6. **Download the trained model**

* You can download the trained model [frcnn_cosmos_output](url) and put it in the folder ``COSMOS_data_tracktor``
---

#### Now you finished dataset pareparing. You folder should like this:
```
ProjectTrafficIntersection.COSMOS.V2
|
|---Tracktor
|
|---COSMOS_data_tracktor
|    |
|    |---detection_dataset
|    |   |---maskrcnn_lablled-bbox-train-val
|    |   |---detection_dataset3
|    |
|    |---Track_data_cosmos
|    |   |---v_and_p
|    |   |---only_v
|    |   |---only_p
|    |
|    |---mask_detect_result
|    |   |---traffic_video_GOPR0589_190720_1324_1454_90sec_calibrated.mp4.txt
|    |   |---traffic_video_GP010589_190720_0310_0440_90sec_calibrated.mp4.txt
|    |   :
|    |   :
|    | 
|    |---frcnn_cosmos_output
|    |   |---faster_rcnn_fpn_training_cosmos
|    |   |---reid
|    |   |---faster_rcnn_fpn_training_cosmos_PV
|    |   |---faster_rcnn_fpn_training_cosmos_P
:
:
```

---
## Training the Faster R-CNN for Tracktor
1. First you should run the command below to install some scripts for training:
```
# Install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# Download TorchVision repo to use some files from
# references/detection
git clone https://github.com/pytorch/vision.git
cd vision

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
```

2. Train the Faster R-CNN to predit pedestrians next location: ``Notebook_experiment/Faster R-CNN Training Origin & Aug_ pedestrian.ipynb``. After that you will get models saved in ``COSMOS_data_tracktor/frcnn_cosmos_output/faster_rcnn_fpn_training_cosmos``

3. Train the Faster R-CNN to predit vehicles next location: ``Notebook_experiment/Faster R-CNN Training Origin & Aug_ vehicle.ipynb``. After that you will get models saved in ``COSMOS_data_tracktor/frcnn_cosmos_output/faster_rcnn_fpn_training_cosmos_P ``

---
## Get the tracking result on CVAT videos

* Look the ``Tracktor/experiments/cfgs/tracktor.yaml`` change some super parameter as you like

* To get the result on 10 CVAT videos, you can run ``python test_tracktor.py``
---
## Do evaluation

---
## Generate the result to see

---
## References
1. Bergmann, P., Meinhardt, T. and Leal-Taixe, L., 2019. Tracking without bells and whistles. In Proceedings of the IEEE International Conference on Computer Vision (pp. 941-951).
2. [phil-bergmann/tracking_wo_bnw](https://github.com/phil-bergmann/tracking_wo_bnw)




































