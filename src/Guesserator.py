import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.utils import custom_object_scope

class CRNNLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, stride, out_channels, rnn_n_layers, rnn_type='lstm', bidirectional=False, **kwargs):
        super(CRNNLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.rnn_n_layers = rnn_n_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

    def build(self, input_shape):
        # Initialize RNN Cells
        rnn_cell = {
            'simple': tf.keras.layers.SimpleRNNCell,
            'lstm': tf.keras.layers.LSTMCell,
            'gru': tf.keras.layers.GRUCell
        }.get(self.rnn_type, tf.keras.layers.SimpleRNNCell)

        rnn_cells = [rnn_cell(self.out_channels) for _ in range(self.rnn_n_layers)]
        if self.bidirectional:
            self.rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(rnn_cells, return_sequences=True))
        else:
            self.rnn_layer = tf.keras.layers.RNN(rnn_cells, return_sequences=True)

        super(CRNNLayer, self).build(input_shape)

    def call(self, inputs):
        # Expand dimensions to be compatible with extract_patches
        inputs_expanded = tf.expand_dims(inputs, axis=2)  # Shape becomes [batch, time_steps, 1, features]

        # Extract patches
        patches = tf.image.extract_patches(
            images=inputs_expanded,
            sizes=[1, self.kernel_size, 1, 1],
            strides=[1, self.stride, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        # Reshape patches to feed into RNN
        batch_size = tf.shape(inputs)[0]
        num_patches = tf.shape(patches)[1]  # This is dynamic based on input size
        patches = tf.reshape(patches, [batch_size, num_patches, self.kernel_size * inputs.shape[-1]])

        # Process patches with RNN
        outputs = self.rnn_layer(patches)

        # Reshape outputs to match expected shape: [batch_size, time_steps_after_stride, output_channels]
        outputs = tf.reshape(outputs, [batch_size, -1, (self.out_channels * (2 if self.bidirectional else 1))])

        return outputs

#Preprocess a single audio file
def load_single_audio_file(file_path):
    num_classes = 10  # There are 10 genres of music to be classified
    features = 160

    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio)  # Decode to waveform
    audio = tf.squeeze(audio, axis=-1)  # Remove last dimension

    # Pad or cut audio to a fixed length of 16000 samples
    audio_length = tf.shape(audio)[0]
    audio = tf.pad(audio, [[0, tf.maximum(0, 16000 - audio_length)]], constant_values=0)  # Pad to 16000 samples
    audio = audio[:16000]  # Cut to 16000 samples if longer

    # Extract Mel spectrogram
    spectrogram = tf.signal.stft(audio, frame_length=1024, frame_step=512)
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=features, num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=16000, lower_edge_hertz=0, upper_edge_hertz=8000
    )
    mel_spec = tf.matmul(tf.abs(spectrogram), mel_spectrogram)

    # Reshape the spectrogram to match the input shape of the model
    mel_spec = tf.expand_dims(mel_spec, axis=0)  # Add batch dimension
    return mel_spec

#Takes the path of a mono wav file and returns a guess for it
def guesserate(file_path):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(["blues", "classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"])

    # Load the model
    with custom_object_scope({'CRNNLayer': CRNNLayer}):
        model_load = tf.keras.models.load_model('good_crnn_model.keras')

    # Load and preprocess the audio file
    preprocessed_audio = load_single_audio_file(file_path)

    # Make a prediction
    predictions = model_load.predict(preprocessed_audio, verbose=0)

    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=-1)

    # Convert the predicted class index back to the genre name
    predicted_genre = label_encoder.inverse_transform(predicted_class)

    return (predicted_genre[0], predictions[0][predicted_class][0])

if __name__ == '__main__':
    print(guesserate("Data/genres_original/blues/blues.00000.wav"))
    print(guesserate("spring.wav"))
