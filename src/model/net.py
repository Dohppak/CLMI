import torch
torch.manual_seed(123)
import torch.nn as nn
import torchaudio
import torchvision.models as models
from .ops import *

class ShortChunkCNN_Res(nn.Module):
    def __init__(self, 
                n_channels: int,
                sample_rate: int,
                n_fft: int,
                f_min: int,
                f_max: int,
                n_mels: int,
                TEST = False):
        super(ShortChunkCNN_Res, self).__init__()
        self.TEST = TEST

        # spec
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.conv1 = Res_2d(1, n_channels, stride=2)
        self.conv2 = Res_2d(n_channels, n_channels, stride=2)
        self.conv3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.conv4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.conv5 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.conv6 = Res_2d(n_channels*2, n_channels*4, stride=2)
        self.conv7 = Res_2d(n_channels*4, n_channels*8, stride=2)

    def forward(self, x):
        x= self.spec_bn(self.to_db(self.spec(x)))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.squeeze(2)
        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)
        return x

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 1:
            x = self.layer(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

class CRMI(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.audio_backbone = ShortChunkCNN_Res(
            n_channels= args.n_channels,
            sample_rate= args.sample_rate,
            n_fft= args.n_fft,
            f_min= args.f_min,
            f_max= args.f_max,
            n_mels= args.n_mels)  
        self.audio_projector = projection_MLP(args.n_channels*8, args.feat_dim, args.num_proj_layers)
        self.audio_encoder = nn.Sequential(
            self.audio_backbone,
            self.audio_projector
        )

        self.image_backbone = models.resnet50(pretrained=args.audio_pretrained)
        out_dim = self.image_backbone.fc.weight.shape[1]
        self.image_backbone.fc = nn.Identity()
        self.image_projector = projection_MLP(out_dim, args.feat_dim, args.num_proj_layers)
        self.image_encoder = nn.Sequential(
            self.image_backbone,
            self.image_projector
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_audio(self, audio):
        return self.audio_encoder(audio)

    def encode_image(self, image):
        return self.image_encoder(image)

    def forward(self, audio, image):
        audio_features = self.encode_audio(audio)
        image_features = self.encode_image(image)
        # normalized features
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits  scale factor 14.2857
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_features @ image_features.t()
        logits_per_image = logit_scale * image_features @ audio_features.t()
        # shape = [global_batch_size, global_batch_size]
        return logits_per_audio, logits_per_image


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["image_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)