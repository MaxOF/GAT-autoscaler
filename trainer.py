import torch
import numpy as np
import time
import os
from prometheus_api_client import PrometheusConnect
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime, timedelta
import time
import math
import pandas as pd
from gat_model import GATAutoScaler
from config import CONFIG

def generate_edge_index():
    edge_sources = []
    edge_targets = []
    service_indices = {service: idx for idx, service in enumerate(CONFIG['services'].keys())}
    
    for service, config in CONFIG['services'].items():
        for dependency in config["dependencies"]:
            edge_sources.append(service_indices[service])
            edge_targets.append(service_indices[dependency])
            edge_sources.append(service_indices[dependency])
            edge_targets.append(service_indices[service])

    if not edge_sources:
        num_services = len(CONFIG['services'])
        for i in range(num_services):
            for j in range(num_services):
                if i != j:
                    edge_sources.append(i)
                    edge_targets.append(j)
    
    return torch.tensor([edge_sources, edge_targets], dtype=torch.long)



prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)


def query_metrics():
    metrics_data = {}
    default_value = 0
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=5)
    step = '15s'

    def process_metric(results, metric_name):
        if not results:
            return

        for item in results:
            pod_or_service = item['metric'].get('container_label_io_kubernetes_pod_name') or item['metric'].get('service')
            service = next((s for s in CONFIG['services'] if s in str(pod_or_service)), None)
            
            if not service:
                continue

            def coalesce(value, default_value):
                if value is None:
                    return default_value
                
                try:
                    num = float(value)
                    if math.isnan(num):
                        return default_value
                    return num
                except (ValueError, TypeError):
                    return value if value != "" else default_value
            
            
            def convert_timestamp(ts):
                return ts * 1000

            if 'values' in item:
                if service not in metrics_data:
                    metrics_data[service] = {}
                
                if metric_name not in metrics_data[service]:
                    metrics_data[service][metric_name] = []
                
                for timestamp, value in item['values']:
                    try:
                        corrected_ts = convert_timestamp(int(float(timestamp)))
                        metrics_data[service][metric_name].append((
                            corrected_ts,
                            coalesce(value, default_value)
                        ))
                    except Exception as e:
                        print(f"Error processing value: {e}")

    queries = {
        'cpu': '''
        sum(rate(container_cpu_usage_seconds_total{container_label_io_kubernetes_pod_name=~"orders-service-.*|payments-service-.*|products-service-.*|rabbitmq",container_label_io_kubernetes_container_name=~".+"}[1m])) 
        by (container_label_io_kubernetes_container_name, container_label_io_kubernetes_pod_name)
        / 
        sum(container_spec_cpu_quota{container_label_io_kubernetes_pod_name=~"orders-service-.*|payments-service-.*|products-service-.*|rabbitmq",container_label_io_kubernetes_container_name=~".+"} 
        / container_spec_cpu_period{container_label_io_kubernetes_pod_name=~"orders-service-.*|payments-service-.*|products-service-.*|rabbitmq",container_label_io_kubernetes_container_name=~".+"}) 
        by (container_label_io_kubernetes_container_name, container_label_io_kubernetes_pod_name) * 100
        ''',
        'memory': '''
        sum(container_memory_usage_bytes{container_label_io_kubernetes_pod_name=~"orders-service-.*|payments-service-.*|products-service-.*|rabbitmq",container_label_io_kubernetes_container_name=~".+"}) 
        by (container_label_io_kubernetes_container_name, container_label_io_kubernetes_pod_name) 
        / 
        sum(container_spec_memory_limit_bytes{container_label_io_kubernetes_pod_name=~"orders-service-.*|payments-service-.*|products-service-.*|rabbitmq",container_label_io_kubernetes_container_name=~".+"} > 0) 
        by (container_label_io_kubernetes_container_name, container_label_io_kubernetes_pod_name) * 100
        ''',
        'rps': 'sum by (service) (rate(requests_total_by_service[1m]))',
        'latency': '''
        histogram_quantile(0.95, 
        sum by (service, route, le) (
            rate(requests_duration_in_seconds_by_service_bucket[1m])
        ))
        '''
    }

    for metric_name, query in queries.items():
        results = prom.custom_query_range(
            query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        process_metric(results, metric_name)

    all_features = []
    for service in CONFIG['services']:
        service_metrics = metrics_data.get(service, {})
        
        timestamps = sorted(set(
            ts for metric in service_metrics.values() 
            for ts, _ in metric
        ))
        
        current_time = int(time.time())
        valid_timestamps = [ts for ts in timestamps if current_time - ts < 3600]

        for ts in valid_timestamps:
            replicas = 1
            features = [
                next((val for t, val in service_metrics.get('cpu', []) if t == ts), 0.0),
                next((val for t, val in service_metrics.get('memory', []) if t == ts), 0.0),
                next((val for t, val in service_metrics.get('rps', []) if t == ts), 0.0),
                next((val for t, val in service_metrics.get('latency', []) if t == ts), 0.0),
                service,
                replicas,
                ts
            ]
            all_features.append(features)
    
    all_features.sort(key=lambda x: x[-1])
 
    return all_features


def prepare_data():
    df = pd.read_csv(
        './metrics_dataset.csv',
        parse_dates=['timestamp'],
        usecols=['timestamp', 'service', 'cpu', 'memory', 'rps', 'latency', 'replicas']
    )

    df = df.dropna(subset=['cpu', 'memory', 'rps', 'latency', 'replicas'])

    latest_time = df['timestamp'].max()
    time_threshold = latest_time - pd.Timedelta(hours=24)
    df = df[df['timestamp'] >= time_threshold]

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    service_dummies = pd.get_dummies(df['service'], prefix='svc')

    numeric_features = ['cpu', 'memory', 'rps', 'replicas', 'hour', 'day_of_week']
    df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce').fillna(0)

    features = pd.concat([df[numeric_features], service_dummies], axis=1)

    features = features.astype(np.float32)

    targets = df['latency'].astype(np.float32)

    scaler = StandardScaler()
    features[numeric_features] = scaler.fit_transform(features[numeric_features])

    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

    return features_tensor, targets_tensor, scaler




def main():
    print("🎓 Учитель запущен (сбор метрик + обучение модели)")

    model = GATAutoScaler(num_features=9)
    edge_index = generate_edge_index()

    features, targets, scaler = prepare_data()


    train_size = int(0.8 * len(features))
    X_train, y_train = features[:train_size], targets[:train_size]
    X_val, y_val = features[train_size:], targets[train_size:]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience = 15
    no_improve = 0
    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        train_data = Data(x=X_train, edge_index=edge_index)
        scale_logits, cpu_scale, mem_scale, attn_weights = model(train_data, return_attention_weights=True)

        pred = mem_scale.squeeze()
        loss = criterion(pred, y_train.float())

        loss += 0.001 * torch.mean(attn_weights ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_data = Data(x=X_val, edge_index=edge_index)
            val_logits, val_cpu, val_mem = model(val_data)
            val_pred = val_mem.squeeze()
            val_loss = criterion(val_pred, y_val.float())
            val_mae = torch.mean(torch.abs(val_pred - y_val))
            val_r2 = 1 - torch.sum((y_val - val_pred) ** 2) / torch.sum((y_val - torch.mean(y_val)) ** 2)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
                  f"Val MAE: {val_mae.item():.2f} | Val R²: {val_r2.item():.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_state': scaler,
                'edge_index': edge_index
            }, CONFIG['model_file'])
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Training finished. Best validation: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
