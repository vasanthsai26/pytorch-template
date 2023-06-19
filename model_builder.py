import torch.nn as nn
import torchvision
import constants

class VehicleImageClassifier(nn.Module):
    """
    Vehicle Image Classifier model based on a backbone network.

    Args:
        image_channels (int): Number of input image channels (default: 3).
        num_classes (int): Number of output classes (default: 10).
    """

    def __init__(
        self,
        image_channels: int = constants.IMAGE_CHANNELS,
        num_classes: int = constants.NUM_CLASSES
    ):
        super(VehicleImageClassifier, self).__init__()

        self.image_channels = image_channels
        self.num_classes = num_classes
        self.classification_model = self.get_pretrained_model()

        self.classification_model.heads = nn.Sequential(
            nn.Linear(in_features=1 * 1 * self.in_features, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        self.initialize_head_weights()

    def forward(self, x):
        """
        Forward pass of the model.

        """
        y_classification = self.classification_model(x)
        return y_classification

    def get_pretrained_model(self):
        """
        Retrieves the backbone network for the classifier.

        """
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        pretrained_model = torchvision.models.vit_b_16(weights=weights)
        for param in pretrained_model.parameters():
            param.requires_grad = False
        self.in_features = pretrained_model.heads.head.in_features
        return pretrained_model
    
    def initialize_head_weights(self):
        """
        Initializes the weights of the network head.
        """
        for module in self.classification_model.heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, 
                                         mode='fan_in', 
                                         nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
