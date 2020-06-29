import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Audio
num_mels = 80
# me
sample_rate = 16000 

# num_freq = 1024
n_fft = 2048
sr = 22050
# frame_length_ms = 50.
# frame_shift_ms = 12.5
preemphasis = 0.97
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
hidden_size = 256
embedding_size = 512
max_db = 100
ref_db = 20
    
n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 2
lr = 0.001
save_step = 1
image_step = 500
batch_size = 32

cleaners='english_cleaners'

data_path = '/home/jupyter/s2s_data/tts/LJSpeech-1.1'
checkpoint_path = '/home/jupyter/s2s_data/models/tts'
sample_path = '/home/jupyter/s2s_data/tts/samples'