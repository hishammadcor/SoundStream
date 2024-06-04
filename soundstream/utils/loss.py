import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

# Adversarial loss for generator
def adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()

    # Calculate the STFT loss
    stft_loss = F.relu(1 - features_stft_disc_G_x[-1]).sum(dim=3).squeeze() / lengths_stft[-1].squeeze()

    # Calculate the waveform discriminator loss for each discriminator
    wave_loss = torch.cat(
        [F.relu(1 - features_wave_disc_G_x[key][-1]).sum(dim=2).squeeze() / lengths_wave[key][-1].squeeze() for key in
         wave_disc_names])

    # Combine STFT and waveform losses and calculate the mean
    loss = torch.cat([stft_loss, wave_loss]).mean()

    return loss

# Feature matching loss
def feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x,
                 lengths_wave, lengths_stft):
    wave_disc_names = lengths_wave.keys()

    # Calculate the STFT feature loss
    stft_loss = torch.stack(
        [((feat_x - feat_G_x).abs().sum(dim=-1) / lengths_stft[i].view(-1, 1, 1)).sum(dim=-1).sum(dim=-1) for
         i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))], dim=1).mean(dim=1,
                                                                                                             keepdim=True)
    # Calculate the waveform feature loss for each discriminator
    wave_loss = torch.stack([torch.stack(
        [(feat_x - feat_G_x).abs().sum(dim=-1).sum(dim=-1) / lengths_wave[key][i] for i, (feat_x, feat_G_x) in
         enumerate(zip(features_wave_disc_x[key], features_wave_disc_G_x[key]))], dim=1) for key in wave_disc_names],
                            dim=2).mean(dim=1)

    # Combine STFT and waveform losses and calculate the mean
    loss = torch.cat([stft_loss, wave_loss], dim=1).mean()

    return loss

# Spectral reconstruction loss
def spectral_reconstruction_loss(x, G_x, eps=1e-4, sr=24000):
    L = 0
    for i in range(6, 12):
        s = 2 ** i
        alpha_s = (s / 2) ** 0.5
        melspec = MelSpectrogram(sample_rate=sr, n_fft=s, hop_length=s // 4, n_mels=8, wkwargs={"device": x.device}).to(
            x.device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)

        # Calculate the spectral reconstruction loss
        loss = (S_x - S_G_x).abs().sum() + alpha_s * (
                    ((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps)) ** 2).sum(dim=-2) ** 0.5).sum()
        L += loss

    return L

# Adversarial loss for discriminator
def adversarial_d_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x,
                       lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()

    # Calculate the real loss for STFT discriminator
    real_stft_loss = F.relu(1 - features_stft_disc_x[-1]).sum(dim=3).squeeze() / lengths_stft[-1].squeeze()

    # Calculate the real loss for waveform discriminators
    real_wave_loss = torch.stack(
        [F.relu(1 - features_wave_disc_x[key][-1]).sum(dim=-1).squeeze() / lengths_wave[key][-1].squeeze() for key in
         wave_disc_names], dim=1)
    real_loss = torch.cat([real_stft_loss.view(-1, 1), real_wave_loss], dim=1).mean()

    # Calculate the generated loss for STFT discriminator
    generated_stft_loss = F.relu(1 + features_stft_disc_G_x[-1]).sum(dim=-1).squeeze() / lengths_stft[-1].squeeze()

    # Calculate the generated loss for waveform discriminators
    generated_wave_loss = torch.stack(
        [F.relu(1 + features_wave_disc_G_x[key][-1]).sum(dim=-1).squeeze() / lengths_wave[key][-1].squeeze() for key in
         wave_disc_names], dim=1)
    generated_loss = torch.cat([generated_stft_loss.view(-1, 1), generated_wave_loss], dim=1).mean()

    return real_loss + generated_loss
