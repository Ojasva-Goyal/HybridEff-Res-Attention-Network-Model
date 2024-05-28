import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50

class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.resnet = resnet50(pretrained=True)
        self.conv = nn.Conv2d(1280, 2048, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        eff_features = self.efficientnet.extract_features(x)
        eff_features = self.conv(eff_features)
        res_features = self.resnet.conv1(x)
        res_features = self.resnet.bn1(res_features)
        res_features = self.resnet.relu(res_features)
        res_features = self.resnet.maxpool(res_features)
        res_features = self.resnet.layer1(res_features)
        res_features = self.resnet.layer2(res_features)
        res_features = self.resnet.layer3(res_features)
        res_features = self.resnet.layer4(res_features)

        combined_features = eff_features + res_features
        attention_weights = self.attention(combined_features)
        attended_features = attention_weights * combined_features

        global_avg_pool = nn.functional.adaptive_avg_pool2d(attended_features, (1, 1))
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.fc(global_avg_pool)
        return output
