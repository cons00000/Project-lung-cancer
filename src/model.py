import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=True, fine_tune=True):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
    model = models.shufflenet_v2_x1_0(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
            
    # change the final classification head, it is trainable,
    # there are 2 classes
    model.fc = nn.Linear(1024, 2)
    return model