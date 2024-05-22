import torch
import torch.nn as nn

class CADModelProcessor(nn.Module):
    def __init__(self):
        super(CADModelProcessor, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*32*32*32, 1024)
        self.fc2 = nn.Linear(1024, 512)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DrawingGenerator(nn.Module):
    def __init__(self):
        super(DrawingGenerator, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, drawing_output_size)  # Размер выходного чертежа зависит от формата данных
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CADToDrawingModel(nn.Module):
    def __init__(self):
        super(CADToDrawingModel, self).__init__()
        self.cad_processor = CADModelProcessor()
        self.drawing_generator = DrawingGenerator()
    
    def forward(self, x):
        features = self.cad_processor(x)
        drawing = self.drawing_generator(features)
        return drawing

# Пример использования
model = CADToDrawingModel()
cad_input = torch.randn((1, 1, 32, 32, 32))  # Пример входной 3D модели
drawing_output = model(cad_input)
print(drawing_output)
