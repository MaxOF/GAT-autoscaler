CONFIG = {
    "prometheus_url": "http://localhost:9090",
    "metrics_interval": 120,
    "training_interval": 86400,
    "data_file": "./metrics_dataset.csv",
    "model_file": "./model_weights.pt",
    "services": {
      "orders-service": {
        "dependencies": ["products-service", "payments-service"],
        "min_cpu": "400m", 
        "max_cpu": "2000m",
        "min_memory": "512Mi",
        "max_memory": "3072Mi"
     },
     "payments-service": {
        "dependencies": [],
        "min_cpu": "400m",
        "max_cpu": "1200m",
        "min_memory": "512Mi",
        "max_memory": "2048Mi"
     },
     "products-service": {
        "dependencies": [],
        "min_cpu": "400m",
        "max_cpu": "1200m",
        "min_memory": "512Mi",
        "max_memory": "2560Mi"
     }
    }
}