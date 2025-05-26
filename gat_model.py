import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GATAutoScaler(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=32, num_classes=3):
        super(GATAutoScaler, self).__init__()
        # Три слоя GAT (как три слоя нейросети)
        self.conv1 = GATConv(num_features, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=2)
        self.conv3 = GATConv(hidden_channels * 2, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
        # Отдельные "головы" для CPU и памяти
        self.cpu_head = torch.nn.Linear(hidden_channels, 1)
        self.mem_head = torch.nn.Linear(hidden_channels, 1)
        
        # Механизм внимания (чтобы учитывать зависимости между сервисами)
        self.dependency_attention = torch.nn.MultiheadAttention(hidden_channels, 4)

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index
        
        # Первый слой GAT
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # Функция активации (как "включение" нейронов)
        x = F.dropout(x, p=0.2, training=self.training)  # Чтобы модель не переобучалась
        
        # Второй слой GAT
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Третий слой GAT
        x = self.conv3(x, edge_index)
        
        # Внимание к зависимостям (анализ, кто на кого влияет)
        x = x.unsqueeze(1)  # Подготовка для внимания
        x, attn_weights = self.dependency_attention(x, x, x)
        x = x.squeeze(1)
        
        # Предсказание количества реплик
        scale_logits = self.lin(x)
        
        # Предсказание CPU и памяти
        cpu_scale = torch.sigmoid(self.cpu_head(x))  # Число от 0 до 1
        mem_scale = torch.sigmoid(self.mem_head(x))  # Число от 0 до 1

        if return_attention_weights:
            return scale_logits, cpu_scale, mem_scale, attn_weights
        
        return scale_logits, cpu_scale, mem_scale