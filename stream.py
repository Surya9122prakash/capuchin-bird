import os
import streamlit as st
import tensorflow as tf
import tensorflow_io as tfio
from itertools import groupby

# Set up GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the trained model
model = tf.keras.models.load_model('model.pkl')

# Helper function to load and preprocess audio files
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# Helper function to load and preprocess MP3 files
def load_mp3_16k_mono(filename):
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

# Helper function to preprocess audio for prediction
def preprocess_mp3(sample, label):  # Add 'label' as the second argument
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label  # Return both the spectrogram and the label


# Define the Streamlit app
def main():
    st.markdown(
        """
        <style>
        [class="appview-container css-1wrcr25 egzxvld6"]{
            background-image: url('https://live.staticflickr.com/7156/6577316685_b1a1b92c59.jpg');
            background-size: cover;
        }
        [class="css-1avcm0n e8zbici2"]{
          visibility: hidden;
        }
        [class="css-10trblm e16nr0p30"]{
            color:black;
            text-align:center;
        }
        [class="css-z8f339 exg6vvm15"]{
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: transparent;
            border-color: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
     
    st.title("Capuchin Bird Population Estimation")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        file_path = "temp.wav"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name.lower().endswith('.mp3'):
            wav = load_mp3_16k_mono(file_path)
            audio_slices = tf.keras.utils.timeseries_dataset_from_array(
                wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1
            )
            audio_slices = audio_slices.map(preprocess_mp3)
            audio_slices = audio_slices.batch(32)

            yhat = model.predict(audio_slices)
            yhat = [1 if prediction > 0.99 else 0 for prediction in yhat]
            calls = tf.math.reduce_sum(yhat).numpy()

            st.success(f"Number of bird calls detected: {calls}")

        elif uploaded_file.name.lower().endswith('.wav'):
            wav = load_wav_16k_mono(file_path)
            audio_slices = tf.keras.utils.timeseries_dataset_from_array(
                wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1
            )
            audio_slices = audio_slices.map(preprocess_mp3, num_parallel_calls=tf.data.AUTOTUNE)
            audio_slices = audio_slices.batch(32)

            yhat = model.predict(audio_slices)
            yhat = [1 if prediction > 0.99 else 0 for prediction in yhat]
            calls = tf.math.reduce_sum([key for key, group in groupby(yhat)]).numpy()

            st.success(f"Number of bird calls detected: {calls}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
