"""
CRNN Model for Bengali Scene Text Recognition
Architecture: CNN (VGG-style) + BiLSTM + CTC
"""
import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM with linear output."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.linear(output)
        return output


class CRNN(nn.Module):
    """
    CRNN: Convolutional Recurrent Neural Network for text recognition.
    
    Architecture:
        - CNN backbone (VGG-style) for feature extraction
        - BiLSTM for sequence modeling
        - Linear layer for character classification
        - CTC for training/decoding
    
    Input: (batch, 1, 32, 128) grayscale images
    Output: (seq_len, batch, num_classes) log probabilities
    """
    
    def __init__(self, num_classes, img_height=32, hidden_size=256, num_lstm_layers=2):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_height = img_height
        self.hidden_size = hidden_size
        
        # CNN Feature Extractor (VGG-style)
        # Input: (batch, 1, 32, 128)
        self.cnn = nn.Sequential(
            # Block 1: 1 → 64 channels
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32→16, 128→64
            
            # Block 2: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16→8, 64→32
            
            # Block 3: 128 → 256 channels (2 conv layers)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 8→4, 32→32
            
            # Block 4: 256 → 512 channels (2 conv layers)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 4→2, 32→32
            
            # Block 5: 512 → 512 channels (final)
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Output: (batch, 512, 1, 31)
        )
        
        # Map from CNN feature channels to LSTM input
        # After CNN: height=1, so we squeeze and treat width as sequence
        self.lstm_input_size = 512
        
        # BiLSTM layers
        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.lstm_input_size, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (batch, 1, H, W) input images
        Returns:
            (seq_len, batch, num_classes) log probabilities
        """
        # CNN features
        conv = self.cnn(x)  # (batch, 512, 1, W')
        
        # Squeeze height dimension and permute for LSTM
        batch, channels, height, width = conv.size()
        assert height == 1, f"CNN output height should be 1, got {height}"
        
        conv = conv.squeeze(2)  # (batch, 512, W')
        conv = conv.permute(0, 2, 1)  # (batch, W', 512) - sequence of feature vectors
        
        # Apply dropout
        conv = self.dropout(conv)
        
        # BiLSTM
        output = self.rnn(conv)  # (batch, W', num_classes)
        
        # Permute to (seq_len, batch, num_classes) for CTC
        output = output.permute(1, 0, 2)  # (W', batch, num_classes)
        
        # Log softmax for CTC
        output = torch.nn.functional.log_softmax(output, dim=2)
        
        return output
    
    def get_seq_length(self, img_width=128):
        """Calculate output sequence length for given input width."""
        # Track width through the network
        w = img_width
        w = w // 2   # MaxPool 1
        w = w // 2   # MaxPool 2
        # MaxPool 3: (2,1) stride - width unchanged 
        # MaxPool 4: (2,1) stride - width unchanged
        w = w - 1     # Conv2d kernel=2 reduces by 1
        return w


def build_model(num_classes, img_height=32, hidden_size=256):
    """Build CRNN model."""
    model = CRNN(num_classes=num_classes, img_height=img_height, hidden_size=hidden_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"CRNN Model Built:")
    print(f"  Num classes (incl. CTC blank): {num_classes}")
    print(f"  Image height: {img_height}")
    print(f"  LSTM hidden size: {hidden_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Output sequence length (W=128): {model.get_seq_length(128)}")
    
    return model


if __name__ == '__main__':
    # Quick test
    model = build_model(num_classes=200, img_height=32)
    dummy = torch.randn(2, 1, 32, 128)
    output = model(dummy)
    print(f"\nTest forward pass:")
    print(f"  Input:  {dummy.shape}")
    print(f"  Output: {output.shape}")
    print(f"  (seq_len={output.shape[0]}, batch={output.shape[1]}, classes={output.shape[2]})")
