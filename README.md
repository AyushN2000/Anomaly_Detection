# Anomaly Detection
In this project, the [Stanford MURA](https://stanfordmlgroup.github.io/competitions/mura/) dataset was explored. It is a large collection of bone X-rays. An algorithm is to be developed to classify whether the test x-ray is abnormal or not.

## Requirements
* **numpy**(v1.17) : It is used for all the matrix operations.
* **tqdm**(v4.45) : It is used to display the progress bar while training occurs
* **matplotlib**(v3.2.1) : This module comes in handy with visualizing the input data and its overall structure
* **jupyter**(v6.0.3) : This module provides a web interface to run python scripts in form of notebooks.
* **opencv-python**(v4.2.0.34) : This module is extensively used in the pre-processsing and post-processsing the data as it is really helpful with functions involving image operations
* **tensorflow**(v2.2.0) : All the deep learning tasks were processed using tensorflow

## Running the project:
1. Clone this repository, and open terminal or CMD prompt at the cloned directory.
1. Create a virtual environment by entering ` python -m venv project_env `
1. Activate the virtual environment by entering `.\project_env\Scripts\activate` for Windows or `source project_env/bin/activate` for macOS/Linux
1. Install the above mentioned required modules from `pip`
1. Download the [dataset](https://stanfordmlgroup.github.io/competitions/mura/) and extract the files to the working directory
1. Run the preproceessing function named as dataset_preprocessor.py by entering the command
  
    `python dataset_preprocessor.py`
   
   This will create a folder with the sub-directories as names of classes
1. Run the Anomaly Detection.ipynb by entering `jupyter notebook Anomaly_Detection.ipynb` and click on 'run all' and wait for the training finishes

1. After completion of training, run the postprocessing script, with filling up the 'path_img' string with the absolute path of the test x-ray image, with the command `python postprocessor.py` 

## Future Scope
This project has a lot of scope for development.
* For instance, the dataset is very noisy, not all images have similar mean of per-image pixel intensities, and the standard deviation from this value is also less than 1 for many images.

A better preprocessing function can be implemented, which takes the present of these noises into account and also that there are some images where the bones are representated with darker shades, where as in most of the studies, bones and surrounding tissues are represented with shades of white.
* The deep learning model has a very high scope of improvement. The  training accuracy yeilded by tthe model before commiting was ~85%. Moreover, the features extracted from the images are not compact and spread out through the image with higher intensities. A better model can be implemented, which extracts features more efficiently and gives a better activation map.
