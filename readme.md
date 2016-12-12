# Machine Learning Engineer Nanodegree
## Project: Distinguish images of dogs from cats

This  project was design for the Capstone Proposal and Capstone Project.

### Getting Started

We developed this project on AWS EC2 instance of type g2.2xlarge. This instance has a high-performance NVIDIA GPU with 1,536 CUDA cores and 4GB of video memory, 15GB of RAM and 8 vCPUs. The machine costs $0.65/hour.
The Operating System installed is Ubuntu 14.04. This insatance has the framework caffe and anaconda2.

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Opencv] (http://opencv.org/)
- [graphviz] (http://www.graphviz.org/)
- [lmdb] (lmdb.tech)
- [pydot](https://pypi.python.org/pypi/pydot)

### Code

The project contains four folders: model, code, input, report.
The **code** folder contains the python code below:
- 'create_lmdb.py'
- 'make_predictions.py'
- 'plot_learning_curve.py'
The **model** folder contains The 
- CNN architecture and its parameters in a configuration file with extension .prototxt
- The solver parameters in a configuration file with extension .prototxt.
- And the generated trained model
- A snapshot of learning curves
- 'submission_model_1.csv' for the kaggle submission.
The **input** folder contains the dataset and data preprocessed.
The **report** folder  contains the proposal.pdf and the report.pdf

### Run

After downloading the train.zip and test1.zip from kaggle,  unzid it on input folder and; next run
**python create_lmdb.py** for data preprocessing
**/home/ubuntu/caffe/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/cats-dogs/input/train_lmdb /home/ubuntu/cats-dogs/input/mean.binaryproto** to generate the mean image of training data.
**/home/ubuntu/caffe/build/tools/caffe train --solver /home/ubuntu/cats-dogs/caffe_model/solver_1.prototxt 2>&1 | tee /home/ubuntu/cats-dogs/caffe_model/model_1_train.log** to train the model.
**python make_predictions_1.py** to make prediction on the test data.

### Data

The cats/dogs data is included as a selection of 25,000 training and testing images collected on data found [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).