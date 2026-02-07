# Kubernetes Introduction

## 1. What is Kubernetes?

Kubernetes (K8s) is a **container orchestration platform**. It automates deployment, scaling, and management of containerized applications.

### Docker vs Kubernetes

| Docker | Kubernetes |
|--------|------------|
| Runs containers | Manages/orchestrates containers |
| Single host | Cluster (multiple servers) |
| Manual scaling | Auto-scaling |
| Simple deployment | Rolling updates, rollbacks |

### Why is Kubernetes needed?

**Problem scenario:**
```
When you have 100 containers...
- Which server should they be deployed to?
- Who restarts containers when they die?
- How to scale when traffic increases?
- Downtime during new version deployment?
```

**Kubernetes solution:**
```
- Auto-scheduling: Deploy to optimal nodes
- Self-healing: Automatic recovery on failure
- Auto-scaling: Scale up/down based on load
- Rolling updates: Zero-downtime deployment
```

---

## 2. Kubernetes Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   Control Plane                         │ │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐ │ │
│  │  │ API     │ │ Scheduler│ │ Controller│ │   etcd    │ │ │
│  │  │ Server  │ │          │ │  Manager  │ │           │ │ │
│  │  └─────────┘ └──────────┘ └───────────┘ └───────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│           ┌────────────────┼────────────────┐               │
│           │                │                │               │
│           ▼                ▼                ▼               │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐          │
│  │   Node 1   │   │   Node 2   │   │   Node 3   │          │
│  │ ┌────────┐ │   │ ┌────────┐ │   │ ┌────────┐ │          │
│  │ │ kubelet│ │   │ │ kubelet│ │   │ │ kubelet│ │          │
│  │ ├────────┤ │   │ ├────────┤ │   │ ├────────┤ │          │
│  │ │  Pod   │ │   │ │  Pod   │ │   │ │  Pod   │ │          │
│  │ │  Pod   │ │   │ │  Pod   │ │   │ │  Pod   │ │          │
│  │ └────────┘ │   │ └────────┘ │   │ └────────┘ │          │
│  └────────────┘   └────────────┘   └────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role |
|-----------|------|
| **API Server** | Central gateway handling all requests |
| **Scheduler** | Decides which Node to place Pods on |
| **Controller Manager** | Maintains desired state (replication, deployment) |
| **etcd** | Cluster state storage |
| **kubelet** | Manages container execution on each Node |
| **kube-proxy** | Network proxy, service load balancing |

---

## 3. Core Concepts

### Pod

- **Smallest deployment unit** in Kubernetes
- Contains one or more containers
- Containers in same Pod share network/storage

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: nginx
      image: nginx:alpine
      ports:
        - containerPort: 80
```

### Deployment

- **Declarative deployment management** of Pods
- Manages replica count (ReplicaSet)
- Supports rolling updates and rollbacks

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3                    # Maintain 3 Pods
  selector:
    matchLabels:
      app: my-app
  template:                      # Pod template
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: nginx
          image: nginx:alpine
          ports:
            - containerPort: 80
```

### Service

- **Network access point** for Pods
- Load balancing
- Provides consistent access even when Pods change

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app                  # Route traffic to Pods with this label
  ports:
    - port: 80                   # Service port
      targetPort: 80             # Pod port
  type: ClusterIP                # Service type
```

### Service Types

| Type | Description |
|------|-------------|
| `ClusterIP` | Accessible only within cluster (default) |
| `NodePort` | External access via Node ports |
| `LoadBalancer` | Connect to cloud load balancer |

---

## 4. Local Environment Setup

### minikube Installation

Tool for running Kubernetes locally.

**macOS:**
```bash
brew install minikube
```

**Windows (Chocolatey):**
```bash
choco install minikube
```

**Linux:**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### minikube Start

```bash
# Start cluster
minikube start

# Check status
minikube status

# Open dashboard
minikube dashboard

# Stop cluster
minikube stop

# Delete cluster
minikube delete
```

### kubectl Installation

CLI tool for communicating with Kubernetes cluster.

**macOS:**
```bash
brew install kubectl
```

**Windows:**
```bash
choco install kubernetes-cli
```

**Verify:**
```bash
kubectl version --client
```

---

## 5. kubectl Basic Commands

### View Resources

```bash
# View all Pods
kubectl get pods

# View all resources
kubectl get all

# Detailed information
kubectl get pods -o wide

# Output in YAML format
kubectl get pod my-pod -o yaml

# Specify namespace
kubectl get pods -n kube-system
```

### Create/Delete Resources

```bash
# Create from YAML file
kubectl apply -f deployment.yaml

# Delete
kubectl delete -f deployment.yaml

# Delete by name
kubectl delete pod my-pod
kubectl delete deployment my-deployment
```

### Detailed Information

```bash
# Resource details
kubectl describe pod my-pod
kubectl describe deployment my-deployment

# View logs
kubectl logs my-pod
kubectl logs -f my-pod              # Real-time

# Access container
kubectl exec -it my-pod -- /bin/sh
```

### Scaling

```bash
# Change replica count
kubectl scale deployment my-deployment --replicas=5
```

---

## 6. Practice Examples

### Example 1: First Pod Execution

```bash
# 1. Run Pod directly
kubectl run nginx-pod --image=nginx:alpine

# 2. Verify
kubectl get pods

# 3. Detailed information
kubectl describe pod nginx-pod

# 4. Check logs
kubectl logs nginx-pod

# 5. Delete
kubectl delete pod nginx-pod
```

### Example 2: Deploy App with Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hello
  template:
    metadata:
      labels:
        app: hello
    spec:
      containers:
        - name: hello
          image: nginxdemos/hello
          ports:
            - containerPort: 80
```

```bash
# 1. Create Deployment
kubectl apply -f deployment.yaml

# 2. Verify
kubectl get deployments
kubectl get pods

# 3. Delete one Pod (verify auto-recovery)
kubectl delete pod <pod-name>
kubectl get pods  # New Pod created

# 4. Scale up
kubectl scale deployment hello-app --replicas=5
kubectl get pods
```

### Example 3: Expose with Service

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hello-service
spec:
  selector:
    app: hello
  ports:
    - port: 80
      targetPort: 80
  type: NodePort
```

```bash
# 1. Create Service
kubectl apply -f service.yaml

# 2. Verify
kubectl get services

# 3. Access on minikube
minikube service hello-service

# Or port forwarding
kubectl port-forward service/hello-service 8080:80
# Access at http://localhost:8080
```

### Example 4: Full Application (Node.js + MongoDB)

**app-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: node-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: node-app
  template:
    metadata:
      labels:
        app: node-app
    spec:
      containers:
        - name: node
          image: node:18-alpine
          command: ["node", "-e", "require('http').createServer((req,res)=>{res.end('Hello K8s!')}).listen(3000)"]
          ports:
            - containerPort: 3000
          env:
            - name: MONGO_URL
              value: "mongodb://mongo-service:27017/mydb"
---
apiVersion: v1
kind: Service
metadata:
  name: node-service
spec:
  selector:
    app: node-app
  ports:
    - port: 80
      targetPort: 3000
  type: NodePort
```

**mongo-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mongo
  template:
    metadata:
      labels:
        app: mongo
    spec:
      containers:
        - name: mongo
          image: mongo:6
          ports:
            - containerPort: 27017
          volumeMounts:
            - name: mongo-storage
              mountPath: /data/db
      volumes:
        - name: mongo-storage
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: mongo-service
spec:
  selector:
    app: mongo
  ports:
    - port: 27017
      targetPort: 27017
```

```bash
# 1. Deploy MongoDB
kubectl apply -f mongo-deployment.yaml

# 2. Deploy Node.js app
kubectl apply -f app-deployment.yaml

# 3. Verify
kubectl get all

# 4. Access
minikube service node-service
```

---

## 7. Rolling Updates

### Apply Update

```bash
# Update image
kubectl set image deployment/hello-app hello=nginxdemos/hello:latest

# Or modify YAML then apply
kubectl apply -f deployment.yaml
```

### Check Update Status

```bash
# Rollout status
kubectl rollout status deployment/hello-app

# History
kubectl rollout history deployment/hello-app
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/hello-app

# Rollback to specific version
kubectl rollout undo deployment/hello-app --to-revision=2
```

---

## 8. ConfigMap and Secret

### ConfigMap - Configuration Data

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_HOST: "db-service"
  LOG_LEVEL: "info"
```

**Use in Deployment:**
```yaml
spec:
  containers:
    - name: app
      envFrom:
        - configMapRef:
            name: app-config
```

### Secret - Sensitive Data

```bash
# Create Secret
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=secret123
```

```yaml
# Create with YAML (requires base64 encoding)
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  username: YWRtaW4=      # echo -n 'admin' | base64
  password: c2VjcmV0MTIz  # echo -n 'secret123' | base64
```

**Use in Deployment:**
```yaml
spec:
  containers:
    - name: app
      env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
```

---

## 9. Namespaces

Logically separate resources.

```bash
# Create namespaces
kubectl create namespace dev
kubectl create namespace prod

# Deploy to specific namespace
kubectl apply -f deployment.yaml -n dev

# Change default namespace
kubectl config set-context --current --namespace=dev
```

---

## Command Summary

| Command | Description |
|---------|-------------|
| `kubectl get pods` | List Pods |
| `kubectl get all` | List all resources |
| `kubectl apply -f file.yaml` | Create/update resource |
| `kubectl delete -f file.yaml` | Delete resource |
| `kubectl describe pod name` | Detailed information |
| `kubectl logs pod-name` | View logs |
| `kubectl exec -it pod -- sh` | Access container |
| `kubectl scale deployment name --replicas=N` | Scale |
| `kubectl rollout status` | Deployment status |
| `kubectl rollout undo` | Rollback |

---

## Recommended Next Learning

1. **Ingress**: HTTP routing, SSL handling
2. **Persistent Volume**: Permanent storage
3. **Helm**: Package manager
4. **Monitoring**: Prometheus, Grafana
5. **Service Mesh**: Istio, Linkerd

### Additional Learning Resources

- [Kubernetes Official Documentation](https://kubernetes.io/docs/)
- [Kubernetes Tutorial](https://kubernetes.io/docs/tutorials/)
- [Play with Kubernetes](https://labs.play-with-k8s.com/)
