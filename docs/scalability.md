# Scalability Architecture - Fraud Detection System

## Overview

This document outlines the production scalability strategy for the FraudShield fraud detection system. The architecture is designed to handle high-throughput real-time transaction scoring while maintaining low latency.

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  TRANSACTION SOURCES                │
                    │   (POS, Mobile App, Web, ATM, Wire Transfer)        │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────────┐
                    │              APACHE KAFKA CLUSTER                    │
                    │  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
                    │  │ Partition │  │ Partition │  │ Partition │       │
                    │  │     0     │  │     1     │  │     2     │       │
                    │  └───────────┘  └───────────┘  └───────────┘       │
                    │        Topic: "transactions.raw"                     │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────────┐
                    │           FRAUD DETECTION SERVICE (K8s)              │
                    │                                                      │
                    │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
                    │  │  Worker 1  │  │  Worker 2  │  │  Worker N  │    │
                    │  │  (FastAPI) │  │  (FastAPI) │  │  (FastAPI) │    │
                    │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │
                    │        │               │               │            │
                    │  ┌─────▼───────────────▼───────────────▼──────┐    │
                    │  │          REDIS CACHE CLUSTER                │    │
                    │  │   (Feature cache, model predictions)       │    │
                    │  └────────────────────────────────────────────┘    │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────────┐
                    │              OUTPUT TOPICS                           │
                    │  ┌─────────────────┐  ┌──────────────────┐         │
                    │  │ "fraud.alerts"  │  │ "transactions.   │         │
                    │  │  (Flagged txns) │  │   scored"        │         │
                    │  └─────────────────┘  └──────────────────┘         │
                    └─────────────────────────────────────────────────────┘
```

## 1. Apache Kafka for Streaming Transactions

### Why Kafka?

- **High throughput**: Handle 100K+ transactions per second
- **Durability**: Persistent message log ensures no transaction is lost
- **Partitioning**: Enables parallel processing across multiple workers
- **Replay**: Re-process historical transactions for model retraining

### Kafka Topic Configuration

```python
# Topic: transactions.raw
KAFKA_CONFIG = {
    "bootstrap.servers": "kafka-broker-1:9092,kafka-broker-2:9092",
    "topic": "transactions.raw",
    "num_partitions": 12,       # One per consumer worker
    "replication_factor": 3,    # High durability
    "retention.ms": 604800000,  # 7 days retention
}
```

### Kafka Consumer Example

```python
from confluent_kafka import Consumer, KafkaError
import json

consumer = Consumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'fraud-detection-group',
    'auto.offset.reset': 'latest'
})
consumer.subscribe(['transactions.raw'])

while True:
    msg = consumer.poll(timeout=1.0)
    if msg is None:
        continue
    
    transaction = json.loads(msg.value())
    result = fraud_model.predict(transaction)
    
    if result['is_fraud']:
        producer.produce('fraud.alerts', json.dumps(result))
    
    producer.produce('transactions.scored', json.dumps(result))
```

## 2. Redis for Caching

### Why Redis?

- **Sub-millisecond latency**: Critical for real-time scoring
- **Feature caching**: Pre-computed features (velocity, card stats) avoid recalculation
- **Rate limiting**: Prevent API abuse
- **Result caching**: Skip re-scoring identical transactions

### Caching Strategy

```python
import redis
import json
import hashlib

redis_client = redis.Redis(host='redis', port=6379, db=0)

# Cache card statistics for fast feature engineering
def get_card_stats(card_id: int) -> dict:
    cache_key = f"card_stats:{card_id}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # Compute from database
    stats = compute_card_statistics(card_id)
    redis_client.setex(cache_key, 3600, json.dumps(stats))  # 1 hour TTL
    return stats

# Cache recent predictions to avoid re-scoring
def get_cached_prediction(transaction: dict) -> dict:
    txn_hash = hashlib.md5(json.dumps(transaction, sort_keys=True).encode()).hexdigest()
    cache_key = f"pred:{txn_hash}"
    
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    return None
```

### Redis Data Structures

| Use Case | Redis Type | Key Pattern | TTL |
|----------|-----------|-------------|-----|
| Card statistics | Hash | `card:{id}:stats` | 1 hour |
| Velocity features | Sorted Set | `card:{id}:txns` | 24 hours |
| Prediction cache | String | `pred:{hash}` | 5 minutes |
| Rate limiting | String | `rate:{ip}` | 1 minute |
| Model metadata | Hash | `model:current` | No expiry |

## 3. Kubernetes Deployment

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-api
spec:
  replicas: 6
  selector:
    matchLabels:
      app: fraud-detection
  template:
    spec:
      containers:
      - name: api
        image: fraudshield:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 4. Model Serving Optimizations

| Optimization | Description | Impact |
|-------------|-------------|--------|
| Model quantization | Convert float64 → float32 | 2x memory reduction |
| Batch inference | Process multiple txns per call | 5-10x throughput |
| ONNX Runtime | Convert model to ONNX format | 2-3x inference speed |
| Feature pre-computation | Cache aggregated features in Redis | 10x feature engineering speed |
| Async processing | Use async FastAPI endpoints | Better concurrency |

## 5. Monitoring & Alerting

- **Prometheus** + **Grafana** for real-time metrics
- Track: prediction latency, fraud rate drift, model accuracy, throughput
- Alert on: latency > 100ms, fraud rate spike, model staleness > 7 days
