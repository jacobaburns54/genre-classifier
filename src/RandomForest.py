import tensorflow as tf 
import tensorflow_decision_forests as tfdf 
import pandas as pd 
import tensorflow_datasets as tfds 
import time
import librosa
import numpy as np


wavFile = input("Enter .wav file name: ")

start = time.time()

y, sr = librosa.load(wavFile, sr=None)

#all of this is just to extract features, full discloser I used ChatGPT to help me with this because I have never used the librosa library before
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_stft_mean_1 = np.mean(chroma_stft, axis=1)
chroma_stft_mean = np.mean(chroma_stft_mean_1)
chroma_stft_var_1 = np.var(chroma_stft, axis=1)
chroma_stft_var = np.mean(chroma_stft_var_1)
rms = librosa.feature.rms(y=y)
rms_mean = np.mean(rms)
rms_var = np.var(rms)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
spectral_centroid_mean = np.mean(spectral_centroid)
spectral_centroid_var = np.var(spectral_centroid)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
spectral_bandwidth_mean = np.mean(spectral_bandwidth)
spectral_bandwidth_var = np.var(spectral_bandwidth)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
rolloff_mean = np.mean(rolloff)
rolloff_var = np.var(rolloff)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
zero_crossing_rate_mean = np.mean(zero_crossing_rate)
zero_crossing_rate_var = np.var(zero_crossing_rate)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfcc_means = np.mean(mfccs, axis=1)
mfcc_vars = np.var(mfccs, axis=1)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
length = librosa.get_duration(y=y, sr=sr)
label = 'unknown_genre'


#put what we just found into dictionary, the keys should exactly match the featured in the training data
data = {
    'filename': wavFile,
    'length': length,
    'chroma_stft_mean': chroma_stft_mean,
    'chroma_stft_var': chroma_stft_var,
    'rms_mean': rms_mean,
    'rms_var': rms_var,
    'spectral_centroid_mean': spectral_centroid_mean,
    'spectral_centroid_var': spectral_centroid_var,
    'spectral_bandwidth_mean': spectral_bandwidth_mean,
    'spectral_bandwidth_var': spectral_bandwidth_var,
    'rolloff_mean': rolloff_mean,
    'rolloff_var': rolloff_var,
    'zero_crossing_rate_mean': zero_crossing_rate_mean,
    'zero_crossing_rate_var': zero_crossing_rate_var,
    'harmony_mean': 0.0,
    'harmony_var': 0.0,
    'perceptr_mean': 0.0,
    'perceptr_var': 0.0,
    'tempo': tempo,
    'mfcc1_mean': mfcc_means[0],
    'mfcc1_var': mfcc_vars[0],
    'mfcc2_mean': mfcc_means[1],
    'mfcc2_var': mfcc_vars[1],
    'mfcc3_mean': mfcc_means[2],
    'mfcc3_var': mfcc_vars[2],
    'mfcc4_mean': mfcc_means[3],
    'mfcc4_var': mfcc_vars[3],
    'mfcc5_mean': mfcc_means[4],
    'mfcc5_var': mfcc_vars[4],
    'mfcc6_mean': mfcc_means[5],
    'mfcc6_var': mfcc_vars[5],
    'mfcc7_mean': mfcc_means[6],
    'mfcc7_var': mfcc_vars[6],
    'mfcc8_mean': mfcc_means[7],
    'mfcc8_var': mfcc_vars[7],
    'mfcc9_mean': mfcc_means[8],
    'mfcc9_var': mfcc_vars[8],
    'mfcc10_mean': mfcc_means[9],
    'mfcc10_var': mfcc_vars[9],
    'mfcc11_mean': mfcc_means[10],
    'mfcc11_var': mfcc_vars[10],
    'mfcc12_mean': mfcc_means[11],
    'mfcc12_var': mfcc_vars[11],
    'mfcc13_mean': mfcc_means[12],
    'mfcc13_var': mfcc_vars[12],
    'mfcc14_mean': mfcc_means[13],
    'mfcc14_var': mfcc_vars[13],
    'mfcc15_mean': mfcc_means[14],
    'mfcc15_var': mfcc_vars[14],
    'mfcc16_mean': mfcc_means[15],
    'mfcc16_var': mfcc_vars[15],
    'mfcc17_mean': mfcc_means[16],
    'mfcc17_var': mfcc_vars[16],
    'mfcc18_mean': mfcc_means[17],
    'mfcc18_var': mfcc_vars[17],
    'mfcc19_mean': mfcc_means[18],
    'mfcc19_var': mfcc_vars[18],
    'mfcc20_mean': mfcc_means[19],
    'mfcc20_var': mfcc_vars[19],
    'label': label
}

#convert to pandas dataframe
df = pd.DataFrame(data)

# get data (training and test) and put into pandas dataframe
train_df = pd.read_csv("Data/features_30_sec.csv") #training data
#test_df = pd.read_csv("Data/features_30_sec_test.csv") #testing data

# then convert into a tf_dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="label")
#test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="label")
user_input_data = tfdf.keras.pd_dataframe_to_tf_dataset(df, label="label")


#create tuner - not using for class demo bc takes too long
#tuner = tfdf.tuner.RandomSearch(num_trials=1, use_predefined_hps=True)

#train model
#model = tfdf.keras.RandomForestModel(tuner=tuner)
model = tfdf.keras.RandomForestModel()
model.fit(train_ds, verbose=2)
#model.summary()

#uncomment if wanting to compare actual vs predicted for the features_30_sec_test.csv dataset
#model.compile(metrics=["accuracy"])
#model.evaluate(test_ds, return_dict=True)
# trying to print out predictions vs actual values
# predict is similar to evaluate, but if we use evaluate we can't look at the labels
#predictions = model.predict(test_ds)
#predicted_labels = tf.argmax(predictions, axis=1) 
dictLabels = { 0: "blues", 1: "classical", 2:"country", 3:"disco", 4:"hiphop", 5:"jazz", 6:"metal", 7:"pop", 8:"reggae", 9:"rock" }
#predicted_labels_names = []

# want to convert the numbers 0-9 to the actual names of the genres
#for x in predicted_labels.numpy():
#	predicted_labels_names.append(dictLabels[x])
#actual_labels = test_df['label'].to_numpy()

# to more easily see which ones it got right and wrong
#correctSum = 0;
#for i in range(len(actual_labels)):
#	print("Actual vs Predicted:", actual_labels[i], predicted_labels_names[i])
#	if(actual_labels[i] == predicted_labels_names[i]):
#		correctSum += 1
#print("\nNumber correctly guessed:", correctSum, "/", len(actual_labels))
#print("Percent:", float(correctSum)/float(len(actual_labels)))


#use our trained model to predict the genre of the user input song
predict = model.predict(user_input_data)

predict_label = tf.argmax(predict, axis=1)
predict_confidence = predict.max(axis = -1)

predict_np = predict_label.numpy()
final_guess = dictLabels[predict_np[0]]

print("Guess:", final_guess)
print("Confidence:", predict_confidence)
print("Elapsed Time:", time.time()-start)


# Export the model to a SavedModel.
#model.save("model/tuner_1_iteration")
#model.save('random_forest_tuned_1.keras')
