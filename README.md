## Simple articles topics classifier that utilizes SVM machine learning algorithm.

Based on [scikit-learn](http://scikit-learn.org/) and Python 3.
Only binary classification is supported in this version.

### Installation and running

1. You need some articles data to train and classify to be stored at `data` directory.
    Classifier assumes that you have 3 subdirectories:
    ```
    data/classifier_train_data/<class_name>/<article>.txt   # ML algorithm training data
    data/classifier_test_data/<class_name>/<article>.txt    # Classified test data
    data/classifier_x_val_data/<class_name>/<article>.txt   # All available articles data for classifier cross-validation
    ```

2. Install NumPy and SciPy. It's highly recommended by NumPy authors to install binaries system-wide.
    So, we won't use virtualenv.
    ```
    sudo apt-get install python3-numpy python3-scipy
    ```

3. Install required Python packages
    ```
    pip3 install -r requirements.txt
    ```

4. Running
    ```
    python3 text_classifier.py
    ```
