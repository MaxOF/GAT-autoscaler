import os
import torch
import csv
import time
from datetime import datetime
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
from torch_geometric.data import Data
import numpy as np
from config import CONFIG
from sklearn.preprocessing import MinMaxScaler
from gat_model import GATAutoScaler
import pandas as pd
import math


class AutoScaler:
    def __init__(self):
        try:
            config.load_incluster_config()
            print("Connected to Kubernetes cluster (in-cluster)")
        except:
            try:
                config.load_kube_config()
                print("Connected using local kubeconfig")
            except Exception as e:
                print(f"Failed to connect to Kubernetes: {e}")
                raise
        self.apps_api = client.AppsV1Api()
        self.core_api = client.CoreV1Api()
        
        prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
        self.prom =PrometheusConnect(url=prometheus_url, disable_ssl=True)
        

        checkpoint = torch.load(CONFIG["model_file"], weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint)
        self.scaler = checkpoint["scaler_state"]
        self.edge_index = checkpoint["edge_index"]
        self.service_order = list(CONFIG["services"].keys())
        
        self.last_action = {}

    def _load_model(self, checkpoint):
        """Загрузка обученной модели"""
        model = GATAutoScaler(num_features=9).to(self.device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def coalesce(self, value, default_value=0):
        if value is None:
            return default_value
        
        try:
            num = float(value)
            if math.isnan(num):
                return default_value
            return num
        except (ValueError, TypeError):
            return value if value != "" else default_value

    def get_current_metrics(self):
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()

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

        metrics = {s: {'cpu': 0, 'memory': 0, 'rps': 0, 'latency': 0} for s in self.service_order}

        for metric_name, query in queries.items():
            results = self.prom.custom_query(query)
            for r in results:
                label = r['metric'].get('container_label_io_kubernetes_pod_name') or r['metric'].get('service')
                if not label:
                    continue
                for s in self.service_order:
                    if s in label:
                        metrics[s][metric_name] = self.coalesce(r['value'][1])

        rows = []
        for s in self.service_order:
            m = metrics[s]

            row = {
                'cpu': m['cpu'],
                'memory': m['memory'],
                'rps': m['rps'],
                'replicas': 1,
                'hour': hour,
                'day_of_week': day_of_week,
                **{f'svc_{name}': 1 if name == s else 0 for name in self.service_order}
            }
            rows.append(row)
       
        df = pd.DataFrame(rows)

        numeric = ['cpu', 'memory', 'rps', 'replicas', 'hour', 'day_of_week']
        df[numeric] = self.scaler.transform(df[numeric])
        return torch.tensor(df.values, dtype=torch.float32)


    def _get_current_replicas(self, service):
        try:
            if not service or not isinstance(service, str):
                print(f"Invalid service name: {service}")
                return 1
                
            if not hasattr(self, 'apps_api'):
                print("Kubernetes API client not initialized")
                return 1
                
            deployment = self.apps_api.read_namespaced_deployment(
                name=service,
                namespace="default"
            )
            
            if not deployment.status:
                print(f"No status available for deployment {service}")
                return 1
                
            return deployment.status.replicas if deployment.status.replicas else 1
            
        except client.exceptions.ApiException as api_e:
            if api_e.status == 404:
                print(f"Deployment {service} not found in namespace 'default'")
            else:
                print(f"Kubernetes API error for {service}: {api_e}")
            return 1
        except Exception as e:
            print(f"Unexpected error getting replicas for {service}: {type(e).__name__} - {str(e)}")
            return 1

    def _save_metrics(self, metrics):
        with open(CONFIG['metrics_file'], 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'service', 'cpu', 'memory', 'rps', 'latency', 'replicas'])
            writer.writeheader()
            for service, values in metrics.items():
                writer.writerow({
                    'timestamp': values['timestamp'],
                    'service': service,
                    'cpu': values['cpu'],
                    'memory': values['memory'],
                    'rps': values['rps'],
                    'latency': values['latency'],
                    'replicas': values['replicas']
                })

    def predict_scaling(self, metrics):
        try:           
            with torch.no_grad():
                scale_logits, cpu_scales, mem_scales = self.model(
                    Data(x=metrics, edge_index=self.edge_index)
                )
                
            return {
                'replicas': scale_logits.argmax(dim=1).tolist(),
                'cpu': cpu_scales.squeeze().tolist(),
                'memory': mem_scales.squeeze().tolist()
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def apply_scaling(self, predictions):
        for i, (service, config) in enumerate(CONFIG['services'].items()):
            target_replicas = predictions['replicas'][i]

            self._scale_horizontal(service, target_replicas)
     
            cpu_scale = predictions['cpu'][i]
            mem_scale = predictions['memory'][i]

            self._scale_vertical(service, cpu_scale, mem_scale)

    def _scale_horizontal(self, service, target_replicas):
        try:
            current = self._get_current_replicas(service)
            if current == target_replicas:
                return
                
            print(f"Scaling {service} from {current} to {target_replicas} replicas")
            patch = {"spec": {"replicas": target_replicas}}
           
            self.apps_api.patch_namespaced_deployment_scale(
                name=service,
                namespace="default",
                body=patch
            )
        except Exception as e:
            print(f"Horizontal scaling failed for {service}: {e}")

    def _scale_vertical(self, service, cpu_scale, mem_scale):
        try:
            min_cpu = self._parse_resource(config['min_cpu'])
            max_cpu = self._parse_resource(config['max_cpu'])
            target_cpu = min_cpu + cpu_scale * (max_cpu - min_cpu)
            
            min_mem = self._parse_resource(config['min_memory'])
            max_mem = self._parse_resource(config['max_memory'])
            target_mem = min_mem + mem_scale * (max_mem - min_mem)
            
            new_cpu = f"{int(target_cpu * 1000)}m"
            new_mem = f"{int(target_mem)}Mi"
            
            patch = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": service,
                                "resources": {
                                    "requests": {"cpu": new_cpu, "memory": new_mem},
                                    "limits": {"cpu": new_cpu, "memory": new_mem}
                                }
                            }]
                        }
                    }
                }
            }
            
            self.apps_api.patch_namespaced_deployment(
                name=service,
                namespace="default",
                body=patch
            )
            
            print(f"Updated resources for {service}: CPU={new_cpu}, Memory={new_mem}")
            
        except Exception as e:
            print(f"Vertical scaling failed for {service}: {e}")

    def _parse_resource(self, resource_str):
        if resource_str.endswith('m'):
            return float(resource_str[:-1]) / 1000
        elif resource_str.endswith('Mi'):
            return float(resource_str[:-2])
        elif resource_str.endswith('Gi'):
            return float(resource_str[:-2]) * 1024
        return float(resource_str)

    def _generate_edge_index(self):
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
            
        return torch.tensor([edge_sources, edge_targets], dtype=torch.long).to(self.device)

    def run(self):
        print("Autoscaler started")
        while True:
            try:
                metrics = self.get_current_metrics()
                
                predictions = self.predict_scaling(metrics)
                if predictions:
                    for i, svc in enumerate(self.service_order):
                        print(f"\n📦 {svc}")
                        print(f"   🔁 Replicas: {predictions['replicas'][i]}")
                        print(f"   🧠 CPU scale (0-1): {predictions['cpu'][i]:.2f}")
                        print(f"   💾 MEM scale (0-1): {predictions['memory'][i]:.2f}")
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                
            time.sleep(CONFIG['check_interval'])

if __name__ == "__main__":
    scaler = AutoScaler()
    scaler.run()