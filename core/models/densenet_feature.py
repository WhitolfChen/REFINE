import torch
from torchvision.models import densenet121

class DenseNetWithFeatures(torch.nn.Module):
    def __init__(self, num_classes=10, weights=None):
        """
        Initialize the DenseNet121 model with feature extraction and output mapping.

        Args:
            pretrained (bool): Whether to load pretrained weights. Default is True.
        """
        super(DenseNetWithFeatures, self).__init__()
        
        # Load the DenseNet121 model
        self.model = densenet121(num_classes=num_classes, weights=weights)
        
        # Extract feature extraction part (all layers except the classifier)
        self.features = self.model.features
        
        # Add the classifier
        self.classifier = self.model.classifier

    def set_model(self, model):
        self.model = model
        self.features = self.model.features
        self.classifier = self.model.classifier

    def from_input_to_features(self, x):
        """
        Map input to feature representation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Feature tensor of shape (N, C', H', W').
        """
        features = self.features(x)
        return features

    def from_features_to_output(self, features):
        """
        Map feature representation to final output.

        Args:
            features (torch.Tensor): Feature tensor of shape (N, C', H', W').

        Returns:
            torch.Tensor: Output tensor of shape (N, num_classes).
        """
        # Apply global average pooling
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        flattened_features = torch.flatten(pooled_features, 1)
        output = self.classifier(flattened_features)
        return output

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
    # Create a DenseNet121 model
    densenet_model = DenseNetWithFeatures(num_classes=10, weights=None)
    x = torch.randn(1, 3, 32, 32)

    # From input to features
    features = densenet_model.from_input_to_features(x)
    print("DenseNet121 Features shape:", features.shape)

    # From features to output
    output = densenet_model.from_features_to_output(features)
    print("DenseNet121 Output shape:", output.shape)
