# mahhh_eaarz_vurry_niiice
**install requirements** with `pip3 install numpy matplotlib librosa pyaudio pynput scikit-learn tensorflow`

# Collecting Data

run `python3 collect_data.py` and start typing and press **ctrl c** to stop and save the recording, by default the data should be atleast 10 seconds long for noise reduction to work.

then run `python3 clean_data.py` to run onset detection to find keypresses, remove the background noise and remove false positives/negatives (the ctrl c key press is eliminated here as false positive), **by default there is a sample of 39 second long audio of me typing 'lavt' 27 times (start with my data for proof of concept).**

# Checking data

run following command to hear what the data sounds like `mpv data/raw_data.wav data/foreground.wav data/background.wav data/divided/*`

run `cat data/labels.npy` to see the labels and `cat data/times.npy` to see labels with corresponding detection times (ground truth) (first keypress time is 0) (with false +ve/-ve removed in cleaning).

if the data looks good, run `i=0;while [[ -f "data/labels${i}.npy" ]];do;((i++));done;mv "data/labels.npy" "data/labels${i}.npy";mv "data/data.npy" "data/data${i}.npy"` to rename previous data to avoid overwriting it and then repeat the data collection step to collect more data.

# Finally adding all the data

run `cd data;python3 combine_all_data.py;cd ..`

# Training Model
run `python3 train_cnn.py` to train the default tensorflow model (architecture defined and training device specified at line 33 of `train_cnn.py`), on default data it gets .68 accuracy (4 keys so .25 random chance), run `tensorboard --logdir=./logs` to see the training metrics.

##### OR

running cross validation with multiple basic ml models trained on this data with command `python3 cross_val.py` returns something like

    The best classifier is: Random Forest
    {'gradient boost': 0.3683982683982684, 'SVM': 0.43506493506493504, 'Random Forest': 0.4528138528138528, 'K-Nearest Neighbors': 0.3683982683982684, 'logistic regression': 0.27575757575757576}

this output is from the default data, any accuracy above 25% implies learning, basically it is learning from default data :)

then edit line 21 in `train_model.py` with the best model returned from cross validation (default:random forest) and run `python3 train_model.py`  to train and save the model in `./model` folders.

# Inference
```diff
- currently seems to have a memory leak, use the next method for now
```
for parallely collecting data and running inference on different cores using the model, run `python3 multi_fucking_processing.py` this script is largely untested as my(sister's) laptop can't handle the load of multiprocessing :(

by default it looks for tf model, make `use_tf_model=False` to use basic ml model.

the script runs a 14 second shifting window on the data which keeps updating every 2 seconds after first waiting for 14 seconds so that the noise reduction doesn't break.

try changing `number_of_noise_reduction_loops=1` and `cosine_similarity_width=[5]` in `./settings.py` file before collecting training data and change the line 17 and 21 in `multi_fucking_processing.py` file accordingly to decrease noise reduction load, now minimum data length is 5 seconds, `max(cosine_similarity_width)` basically.

try making `me_too_poor=True` in line 16 in `multi_fucking_processing.py` so that script kills the data collecting core after 14 seconds of collecting data.

# Inference for poor kids like me

run `cd predict` then `python3 collect_unlabeled_data.py` to collect unlabelled data (atleast 10 seconds for default noise reduction settings), press **ctrl c** to stop and save, then run `python3 clean_unlabeled_data.py` to clean the data.

by default it looks for tf model, make `use_tf_model=False` in `predict.py` file to use basic ml model, now run `python3 predict.py`  to predict on the unlabelled data.

# Future Improvements

have been trying out different noise reduction techniques with no luck, will keep trying, current one is from [librosa vocal reduction](https://librosa.org/librosa_gallery/auto_examples/plot_vocal_separation.html) so there should not be any human voices in data for now.

spelling checking and word prediction for predictable text (not passwds)

with context guessing + osinting open assistant would be a killer, have been trying to align it, will see.




