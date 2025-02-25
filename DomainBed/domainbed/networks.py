import torch
import torch.nn as nn
import torch.nn.functional as F
import torchFOLDER.models

from domainbed.lib import wide_resnet
import copy

from transformers import BertForSequenceClassification, BertModel


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchFOLDER.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchFOLDER.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams

        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class ModelLoader(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super(ModelLoader, self).__init__()
        """
        model_type: str, type of model to load ('resnet18', 'resnet50', 'densenet121', 'convnext_tiny')
        input_shape: tuple, the shape of the input (should include the transfer flag as last element)
        hparams: dict, hyperparameters such as number of output classes
        """
        self.model_type = hparams['model_arch']
        self.input_shape = input_shape
        self.transfer = hparams['transfer']  # Assuming transfer is the last element in input_shape
        self.num_classes = hparams.get('num_classes', 2)
        self.network = self._load_model()

        # Freeze BatchNorm layers
        self.freeze_bn()

        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def _load_model(self):
        if self.model_type == 'resnet18':
            model = torchFOLDER.models.resnet18(pretrained=True)
            self.n_outputs = 512
        elif self.model_type == 'resnet50':
            model = torchFOLDER.models.resnet50(pretrained=True)
            self.n_outputs = 2048
        elif self.model_type == 'densenet121':
            model = torchFOLDER.models.densenet121(pretrained=True)
            self.n_outputs = 1024
        elif self.model_type == 'convnext_tiny':
            model = torchFOLDER.models.convnext_tiny(pretrained=True)
            self.n_outputs = 768
        elif self.model_type == 'vit_b_16':
            model = torchFOLDER.models.vit_b_16(pretrained=True)
            self.n_outputs =768
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Modify input channels if needed
        nc = self.input_shape[0]
        if nc != 3:
            tmp = model.conv1.weight.data.clone()
            model.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False
            )
            for i in range(nc):
                model.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # Replace final classification layers with Identity
        if 'resnet' in self.model_type:
            model.fc = nn.Identity()
        elif 'densenet' in self.model_type:
            model.classifier = nn.Identity()
        elif 'convnext' in self.model_type:
            model.classifier[2] = nn.Identity()
        elif 'vit' in self.model_type:
            model.heads.head = nn.Identity()

        # Freeze layers for transfer learning
        if self.transfer:
            for param in model.parameters():
                param.requires_grad = False

        return model

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters.
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        """Freeze all BatchNorm2d layers."""
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_model(self):
        return self.model

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ModelLoader(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class Bert(torch.nn.Module):
    """BERT with pre-trained weights"""
    def __init__(self, input_shape, hparams):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.n_outputs = self.bert.config.hidden_size

    def forward(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        # Use the pooled output for classification tasks
        pooled_output = outputs.pooler_output
        return self.dropout(pooled_output)
