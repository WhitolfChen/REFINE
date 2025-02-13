import torch
from torchvision.models import resnet18, resnet50

class ResNetWithFeatures(torch.nn.Module):
    def __init__(self, model_type, num_classes, weights=False):
        """
        Initialize a class that supports different ResNet versions.

        Args:
            model_type (str): "ResNet18" or "ResNet50". Default is "ResNet18".
            pretrained (bool): Whether to load pretrained weights. Default is True.
        """
        super(ResNetWithFeatures, self).__init__()
        
        # Choose ResNet model based on model_type
        if model_type == "ResNet18":
            self.model = resnet18(weights=weights, num_classes=num_classes)
        elif model_type == "ResNet50":
            self.model = resnet50(weights=weights, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'ResNet18' or 'ResNet50'.")
        
        # Extract the feature extraction part (from conv1 to layer4)
        self.features = torch.nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )
        
        # Add the post-processing layers
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc

    def set_model(self, model):
        self.model = model
        self.features = torch.nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc
        
    def from_input_to_features(self, x):
        """
        Map input to feature representation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Feature tensor of shape (N, C', H', W').
        """
        return self.features(x)

    def from_features_to_output(self, features):
        """
        Map feature representation to final output.

        Args:
            features (torch.Tensor): Feature tensor of shape (N, C', H', W').

        Returns:
            torch.Tensor: Output tensor of shape (N, num_classes).
        """
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        """
        Perform the full forward pass from input to output.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, num_classes).
        """
        features = self.from_input_to_features(x)
        output = self.from_features_to_output(features)
        return output

if __name__ == "__main__":
    # Usage example
    # Create a ResNet18 model
    ResNet18_model = ResNetWithFeatures(model_type="ResNet18", num_classes=10, weights=None)
    x = torch.randn(1, 3, 224, 224)
    features18 = ResNet18_model.from_input_to_features(x)
    output18 = ResNet18_model.from_features_to_output(features18)
    print("ResNet18 Features shape:", features18.shape)
    print("ResNet18 Output shape:", output18.shape)

    # Create a ResNet50 model
    ResNet50_model = ResNetWithFeatures(model_type="ResNet50", num_classes=10, weights=None)
    features50 = ResNet50_model.from_input_to_features(x)
    output50 = ResNet50_model.from_features_to_output(features50)
    print("ResNet50 Features shape:", features50.shape)
    print("ResNet50 Output shape:", output50.shape)
