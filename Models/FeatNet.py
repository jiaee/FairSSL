
import torch.nn as nn


class EEGFeatNet(nn.Module):
    def __init__(self, input_shape=(128, 500), num_filters=[64, 128, 256, 512], kernel_sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2], padding=[1, 1, 1, 1]):
        super(EEGFeatNet, self).__init__()

        self.layers = nn.ModuleList()
        in_channels = 1
        for i, out_channels in enumerate(num_filters):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=(kernel_sizes[i], kernel_sizes[i]),
                        stride=(strides[i], strides[i]),
                        padding=(padding[i], padding[i]),
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            )
            in_channels = out_channels

        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters[-1], 512, bias=False)
        self.fc2 = nn.Linear(512, 256) 

    def normalize_input(self, x):
        """
        Normalize input data over the time dimension for each data point in the batch.
        """
        mean = x.mean(dim=3, keepdim=True)
        std = x.std(dim=3, keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            pass              
        elif x.dim() == 5 and x.size(2) == 1:
            x = x.squeeze(2)
        else:
            raise ValueError(f"EEGFeatNet got unexpected shape: {tuple(x.shape)}")

        # x = self.normalize_input(x)

        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = torch.relu(x)
        x = self.fc2(x)
        return x


class AudioFeatNet(nn.Module):
    def __init__(self, input_shape=(512, 313), num_filters=[64, 128, 256, 512], kernel_sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2], padding=[1, 1, 1, 1]):
        super(AudioFeatNet, self).__init__()

        self.layers = nn.ModuleList()
        in_channels = 1
        for i, out_channels in enumerate(num_filters):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                              kernel_size=(kernel_sizes[i], kernel_sizes[i]), 
                              stride=(strides[i], strides[i]), 
                              padding=(padding[i], padding[i]), 
                              bias=False
                              ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            )
            in_channels = out_channels

        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1))) 

        self.fc1 = nn.Linear(num_filters[-1], 512, bias=False)
        self.fc2 = nn.Linear(512, 256) 

    def normalize_input(self, x):
        """
        Normalize input data over the time dimension for each data point in the batch.
        """
        mean = x.mean(dim=3, keepdim=True) 
        std = x.std(dim=3, keepdim=True) + 1e-6 
        return (x - mean) / std

    def forward(self, x):
        # x = self.normalize_input(x)

        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = torch.relu(x)
        x = self.fc2(x)
        return x
    

class MultiModalCNN(nn.Module):
    def __init__(self, cnn_audio, cnn_eeg):
        super(MultiModalCNN, self).__init__()
        self.cnn_audio = cnn_audio
        self.cnn_eeg = cnn_eeg

    def forward(self, audio_inputs, eeg_inputs):
        z_waveform = self.cnn_audio(audio_inputs)
        z_spectrogram = self.cnn_eeg(eeg_inputs)
        return z_waveform, z_spectrogram
    
