## Using
train.py for training models

Parameter introduction:
- save dir: string type, setting the storage address for training and validation results
- note: creating empty folders is not supported, so it is necessary to create the folders in advance and set them up when running on your own
- weights path: string type, pretrained model address
- data: The dataset path is set using the yaml file, referring to data.yaml in the config folder
- hyp: the same as above

The other parameters can be determined by their literal meaning

predict.py for prediction
-The parameter settings are shown above

