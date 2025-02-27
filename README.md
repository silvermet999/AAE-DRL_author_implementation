This repository is the final version of AAE-DRL.

# Data Processing
We used the UNSW-NB15 dataset, the final version of the dataset (joined, cleaned and preprocessed) is available in data/prep.zip

# 1/ AAE training and testing on unaugmented data
To train the AAE model on the unaugmented dataset, run:
```bash
python AAE_main.py --train --unaug_dataset
```
To test the AAE model on the unaugmented dataset, run:
```bash
python AAE_main.py --unaug_dataset
```

# 2/ Benchmark classification of unaugmented data
To evaluate the generated data on Gradient Boosting, run:
```bash
python benchmark_clf_main.py --unaug_dataset
```
To evaluate the generated data on the other benchmark classifiers, set one of the following flags to True:
--xgb_clf : to use Extreme Gradient Boosting
--KNN_clf : to use K Nearest Neighbor
--rf_clf : to use Random Forest
Example:
```bash
python benchmark_clf_main.py --rf_clf=True --unaug_dataset
```
On another note, we provided the code for the Optuna trials in the clfs/clf_optim.py file.

# 3/ TabNet classifier pre-training and label prediction
To pre-train the classifier, run the original (supervised) dataset that was used to train the AAE:
```bash
python classifier_main.py --train --unaug_dataset
```
Add --label_gen flag to generate labels for synthetic (unsupervised and unaugmented) dataset:
```bash
python classifier_main.py --train --unaug_dataset --label_gen
```
To test the classifier on unaugmented dataset, run:
```bash
python classifier_main.py --unaug_dataset
```

# 4/ DRL training and testing
To train the DRL algorithm and generate new synthetic data, run:
```bash
python DRL_main.py --train
```
To test the DRL algorithm, run:
```bash
python DRL_main.py
```

# 5/ TabNet classifier label prediction
To predict labels for the new synthetic dataset, run
```bash
python classifier_main.py --label_gen --synth_dataset_path==<path_to_dataset_generated_in_STEP4>
```

# 6/ AAE training on augmented data
To train the AAE model on the augmented dataset, run:
```bash
python AAE_main.py --train --X_ds==<path_to_dataset_generated_in_STEP4> --y_ds==<path_to_dataset_generated_in_STEP5>
```

# 7/ Benchmark classification on augmented dataset
To evaluate the new generated data on Gradient Boosting, run:
```bash
python benchmark_clf_main.py
```
Similarly to STEP 2, you can change to another classifier using the mentioned flags.
Example:
```bash
python benchmark_clf_main.py --KNN_clf=True
```
