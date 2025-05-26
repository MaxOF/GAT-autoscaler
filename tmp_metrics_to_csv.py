import csv
import numpy as np
import time
import os
from prometheus_api_client import PrometheusConnect
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from kubernetes import client, config
import yaml
from datetime import datetime, timedelta
import time
import math
import pandas as pd
from config import CONFIG

# === 4. ЗАПРОС МЕТРИК ИЗ PROMETHEUS ===
# Prometheus — это система мониторинга, которая собирает данные о нагрузке.
prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)


def query_metrics():
    metrics_data = {}
    default_value = 0
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=5)
    step = '15s'  # 15-секундные интервалы

    # Универсальный обработчик для всех метрик
    def process_metric(results, metric_name):
        if not results:
            return

        for item in results:
            # Определяем имя сервиса
            pod_or_service = item['metric'].get('container_label_io_kubernetes_pod_name') or item['metric'].get('service')
            service = next((s for s in CONFIG['services'] if s in str(pod_or_service)), None)
            
            if not service:
                continue

            def coalesce(value, default_value):
                # Проверяем сначала None
                if value is None:
                    return default_value
                
                # Пытаемся преобразовать в float, если это возможно
                try:
                    num = float(value)
                    if math.isnan(num):  # Проверяем на NaN после конвертации
                        return default_value
                    return num  # Возвращаем число, если не NaN
                except (ValueError, TypeError):
                    # Если value не число (например, строка), возвращаем как есть
                    return value if value != "" else default_value  # Опционально: замена пустой строки
            
            
            def convert_timestamp(ts):
                return ts * 1000


            # Сохраняем ВСЕ значения временного ряда
            if 'values' in item:
                if service not in metrics_data:
                    metrics_data[service] = {}
                
                if metric_name not in metrics_data[service]:
                    metrics_data[service][metric_name] = []
                
               # Обрабатываем каждое значение с коррекцией timestamp
                for timestamp, value in item['values']:
                    try:
                        corrected_ts = convert_timestamp(int(float(timestamp)))
                        # replicas = get_current_replicas(service)
                        metrics_data[service][metric_name].append((
                            corrected_ts,
                            coalesce(value, default_value)
                        ))
                    except Exception as e:
                        print(f"Error processing value: {e}")

    # Запросы метрик
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

    # Выполняем все запросы
    for metric_name, query in queries.items():
        results = prom.custom_query_range(
            query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        process_metric(results, metric_name)

    # Формируем массив всех значений
    all_features = []
    for service in CONFIG['services']:
        service_metrics = metrics_data.get(service, {})
        
        # Собираем все временные точки
        timestamps = sorted(set(
            ts for metric in service_metrics.values() 
            for ts, _ in metric
        ))
        
         # Фильтруем слишком старые метки
        current_time = int(time.time())
        valid_timestamps = [ts for ts in timestamps if current_time - ts < 3600]  # До 1 часа

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
    
    # Сортируем по времени и преобразуем в numpy array
    all_features.sort(key=lambda x: x[-1])  # Сортировка по timestamp
 
    return all_features


def save_to_csv():
        """Дописывает метрики в существующий CSV файл с помощью pandas"""
        try:
            # Получаем новые метрики
            new_metrics = query_metrics()
            
            if not new_metrics:
                print("Нет новых метрик для сохранения")
                return
                
            # Создаем DataFrame из новых данных
            new_df = pd.DataFrame(new_metrics, columns=[
                'cpu', 'memory', 'rps', 'latency', 'service', 'replicas', 'timestamp'
            ])
           
            # Конвертируем timestamp в datetime
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            
            # Проверяем существование файла
            if os.path.exists(CONFIG['data_file']):
                # Читаем существующие данные
                existing_df = pd.read_csv(
                    CONFIG['data_file'],
                    parse_dates=['timestamp']
                )
         
                # Объединяем старые и новые данные
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
      
                # Удаляем возможные дубликаты
                combined_df = combined_df.drop_duplicates(
                    subset=['timestamp', 'service'],
                    keep='last'
                )
              
            else:
                combined_df = new_df
            
            # Сохраняем обратно в CSV
            combined_df.to_csv(
                CONFIG['data_file'],
                index=False,
                columns=['timestamp', 'service', 'cpu', 'memory', 'rps', 'latency', 'replicas']
            )
            
            print(f"Успешно сохранено {len(new_df)} новых записей. Всего записей: {len(combined_df)}")
            
        except Exception as e:
            print(f"Ошибка при сохранении метрик: {str(e)}")


save_to_csv()