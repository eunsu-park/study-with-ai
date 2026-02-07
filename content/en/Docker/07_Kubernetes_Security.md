# 07. Kubernetes Security

## Learning Objectives
- Understand Kubernetes security architecture
- Implement access control with RBAC
- Network isolation with NetworkPolicy
- Manage Secrets and sensitive information
- Apply Pod security policies

## Table of Contents
1. [Kubernetes Security Overview](#1-kubernetes-security-overview)
2. [RBAC (Role-Based Access Control)](#2-rbac-role-based-access-control)
3. [ServiceAccount](#3-serviceaccount)
4. [NetworkPolicy](#4-networkpolicy)
5. [Secrets Management](#5-secrets-management)
6. [Pod Security](#6-pod-security)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. Kubernetes Security Overview

### 1.1 4C Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Cluster                              │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              Container                       │   │   │
│  │  │  ┌─────────────────────────────────────┐   │   │   │
│  │  │  │            Code                      │   │   │   │
│  │  │  │  - Vulnerability scanning            │   │   │   │
│  │  │  │  - Dependency management             │   │   │   │
│  │  │  │  - Secure coding                     │   │   │   │
│  │  │  └─────────────────────────────────────┘   │   │   │
│  │  │  - Image security                           │   │   │
│  │  │  - Runtime security                         │   │   │
│  │  │  - Resource limits                          │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │  - RBAC, NetworkPolicy                            │   │
│  │  - Secrets management                             │   │
│  │  - Pod security                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│  - Network security                                        │
│  - IAM, firewall                                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Authentication and Authorization

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    User     │────▶│   API Server │────▶│  Resources  │
│  (kubectl)  │     │              │     │   (Pods)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  AuthN   │ │  AuthZ   │ │ Admission│
        │          │ │          │ │ Control  │
        ├──────────┤ ├──────────┤ ├──────────┤
        │• Certs   │ │• RBAC    │ │• Validate│
        │• Tokens  │ │• ABAC    │ │• Mutate  │
        │• OIDC    │ │• Webhook │ │• Policy  │
        └──────────┘ └──────────┘ └──────────┘
```

### 1.3 Security Components

```yaml
# Check current cluster security status
# Check API server settings
kubectl describe pod kube-apiserver-<master-node> -n kube-system

# Check authentication mode
kubectl api-versions | grep rbac
# rbac.authorization.k8s.io/v1

# Check cluster permissions
kubectl auth can-i --list
```

---

## 2. RBAC (Role-Based Access Control)

### 2.1 RBAC Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                      RBAC Components                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐                  ┌───────────────┐      │
│  │     Role      │                  │  ClusterRole  │      │
│  │  (Namespace)  │                  │   (Cluster)   │      │
│  └───────┬───────┘                  └───────┬───────┘      │
│          │                                  │               │
│          │ Binding                          │ Binding       │
│          ▼                                  ▼               │
│  ┌───────────────┐                  ┌───────────────┐      │
│  │ RoleBinding   │                  │ClusterRole    │      │
│  │               │                  │   Binding     │      │
│  └───────┬───────┘                  └───────┬───────┘      │
│          │                                  │               │
│          └──────────────┬───────────────────┘               │
│                         ▼                                   │
│                 ┌───────────────┐                           │
│                 │   Subjects    │                           │
│                 │ • User        │                           │
│                 │ • Group       │                           │
│                 │ • ServiceAcc  │                           │
│                 └───────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Role Definition

```yaml
# role-pod-reader.yaml
# Pod read permission in specific namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: pod-reader
rules:
- apiGroups: [""]          # "" = core API group
  resources: ["pods"]
  verbs: ["get", "watch", "list"]

---
# role-deployment-manager.yaml
# Deployment management permission
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: deployment-manager
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]

---
# role-secret-reader.yaml
# Read specific Secrets only (using resourceNames)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: specific-secret-reader
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["app-config", "db-credentials"]  # Specific resources only
  verbs: ["get"]
```

### 2.3 ClusterRole Definition

```yaml
# clusterrole-node-reader.yaml
# Read node information across cluster
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: node-reader
rules:
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "watch", "list"]

---
# clusterrole-pv-manager.yaml
# PersistentVolume management (cluster-scoped resource)
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pv-manager
rules:
- apiGroups: [""]
  resources: ["persistentvolumes"]
  verbs: ["get", "list", "watch", "create", "delete"]
- apiGroups: [""]
  resources: ["persistentvolumeclaims"]
  verbs: ["get", "list", "watch"]

---
# clusterrole-namespace-admin.yaml
# Admin role across all namespaces
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: namespace-admin
rules:
- apiGroups: [""]
  resources: ["namespaces"]
  verbs: ["get", "list", "watch", "create", "delete"]
- apiGroups: [""]
  resources: ["*"]
  verbs: ["*"]

---
# Aggregated ClusterRole
# clusterrole-monitoring.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: monitoring
  labels:
    rbac.example.com/aggregate-to-monitoring: "true"
aggregationRule:
  clusterRoleSelectors:
  - matchLabels:
      rbac.example.com/aggregate-to-monitoring: "true"
rules: []  # Rules are automatically aggregated
```

### 2.4 RoleBinding & ClusterRoleBinding

```yaml
# rolebinding-pod-reader.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: development
subjects:
- kind: User
  name: jane
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io

---
# rolebinding-sa.yaml
# Bind role to ServiceAccount
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-deployment-binding
  namespace: development
subjects:
- kind: ServiceAccount
  name: app-deployer
  namespace: development
roleRef:
  kind: Role
  name: deployment-manager
  apiGroup: rbac.authorization.k8s.io

---
# clusterrolebinding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: node-reader-binding
subjects:
- kind: Group
  name: ops-team
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: node-reader
  apiGroup: rbac.authorization.k8s.io

---
# Bind ClusterRole to specific namespace
# (Reuse ClusterRole)
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: admin-binding
  namespace: staging
subjects:
- kind: User
  name: admin-user
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole      # ClusterRole but
  name: admin            # Scope limited by RoleBinding
  apiGroup: rbac.authorization.k8s.io
```

### 2.5 RBAC Testing and Debugging

```bash
# Check permissions
kubectl auth can-i create pods --namespace development
# yes

kubectl auth can-i delete pods --namespace production --as jane
# no

kubectl auth can-i '*' '*' --all-namespaces --as system:serviceaccount:default:admin
# yes

# Check all permissions for specific user
kubectl auth can-i --list --as jane --namespace development

# View RBAC resources
kubectl get roles -n development
kubectl get rolebindings -n development
kubectl get clusterroles
kubectl get clusterrolebindings

# Detailed information
kubectl describe role pod-reader -n development
kubectl describe rolebinding read-pods -n development
```

---

## 3. ServiceAccount

### 3.1 ServiceAccount Basics

```yaml
# serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: production
  annotations:
    description: "Application service account for production"
# Tokens are not automatically created in Kubernetes 1.24+

---
# Token creation (Kubernetes 1.24+)
apiVersion: v1
kind: Secret
metadata:
  name: app-sa-token
  namespace: production
  annotations:
    kubernetes.io/service-account.name: app-service-account
type: kubernetes.io/service-account-token
```

### 3.2 Using ServiceAccount in Pods

```yaml
# pod-with-sa.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
  namespace: production
spec:
  serviceAccountName: app-service-account
  automountServiceAccountToken: true  # Auto-mount token
  containers:
  - name: app
    image: myapp:latest
    # Token mounted at /var/run/secrets/kubernetes.io/serviceaccount/

---
# Disable token mount (security hardening)
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  serviceAccountName: restricted-sa
  automountServiceAccountToken: false  # Do not mount token
  containers:
  - name: app
    image: myapp:latest
```

### 3.3 RBAC for ServiceAccount

```yaml
# ServiceAccount for CI/CD pipeline example
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cicd-deployer
  namespace: cicd

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cicd-deployer-role
rules:
# Deployment management
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Service management
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# ConfigMap, Secret read
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
# Pod status check
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cicd-deployer-binding
subjects:
- kind: ServiceAccount
  name: cicd-deployer
  namespace: cicd
roleRef:
  kind: ClusterRole
  name: cicd-deployer-role
  apiGroup: rbac.authorization.k8s.io
```

### 3.4 Using ServiceAccount Tokens

```bash
# Get ServiceAccount token
TOKEN=$(kubectl create token app-service-account -n production)

# Or get from Secret
TOKEN=$(kubectl get secret app-sa-token -n production -o jsonpath='{.data.token}' | base64 -d)

# Call API with token
curl -k -H "Authorization: Bearer $TOKEN" \
  https://kubernetes.default.svc/api/v1/namespaces/production/pods

# Create kubeconfig
kubectl config set-credentials sa-user --token=$TOKEN
kubectl config set-context sa-context --cluster=my-cluster --user=sa-user
```

---

## 4. NetworkPolicy

### 4.1 NetworkPolicy Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NetworkPolicy Behavior                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Without NetworkPolicy:                                     │
│  ┌─────┐     ┌─────┐     ┌─────┐                           │
│  │Pod A│◀───▶│Pod B│◀───▶│Pod C│  All traffic allowed      │
│  └─────┘     └─────┘     └─────┘                           │
│                                                             │
│  With NetworkPolicy:                                        │
│  ┌─────┐     ┌─────┐     ┌─────┐                           │
│  │Pod A│────▶│Pod B│  ✗  │Pod C│  Restricted by policy    │
│  └─────┘     └─────┘     └─────┘                           │
│                                                             │
│  ⚠️  Note: CNI plugin must support NetworkPolicy           │
│      (Calico, Cilium, Weave Net, etc.)                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Basic NetworkPolicy

```yaml
# deny-all-ingress.yaml
# Deny all inbound traffic by default
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: production
spec:
  podSelector: {}  # Apply to all Pods
  policyTypes:
  - Ingress
  # No ingress rules = deny all inbound

---
# deny-all-egress.yaml
# Deny all outbound traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-egress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Egress
  # No egress rules = deny all outbound

---
# default-deny-all.yaml
# Deny all traffic (most restrictive)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

### 4.3 Allow Policies

```yaml
# allow-frontend-to-backend.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080

---
# allow-backend-to-database.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-database
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: database
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend
    ports:
    - protocol: TCP
      port: 5432

---
# Allow access from another namespace
# allow-from-monitoring.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-monitoring
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
      podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 9090
```

### 4.4 Complex Policies

```yaml
# comprehensive-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-server-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # 1. Allow from frontend in same namespace
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 443
  # 2. Allow from Ingress Controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 443
  # 3. Allow from specific IP range
  - from:
    - ipBlock:
        cidr: 10.0.0.0/8
        except:
        - 10.0.1.0/24  # Exclude this range
    ports:
    - protocol: TCP
      port: 443
  egress:
  # 1. Outbound to database
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  # 2. Outbound to cache server
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  # 3. Allow DNS (required!)
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
```

### 4.5 NetworkPolicy Debugging

```bash
# View NetworkPolicy
kubectl get networkpolicy -n production
kubectl describe networkpolicy api-server-policy -n production

# Check Pod labels
kubectl get pods -n production --show-labels

# Connection test
kubectl run test-pod --rm -it --image=busybox -n production -- /bin/sh
# Inside Pod
wget -qO- --timeout=2 http://backend-service:8080
nc -zv database-service 5432

# Check CNI plugin
kubectl get pods -n kube-system | grep -E "calico|cilium|weave"
```

---

## 5. Secrets Management

### 5.1 Secret Types

```yaml
# 1. Opaque (generic data)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: production
type: Opaque
data:
  # base64 encoding required
  username: YWRtaW4=         # admin
  password: cGFzc3dvcmQxMjM=  # password123
stringData:
  # stringData doesn't need encoding
  api-key: my-secret-api-key

---
# 2. kubernetes.io/dockerconfigjson (container registry)
apiVersion: v1
kind: Secret
metadata:
  name: docker-registry-secret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: eyJhdXRocyI6eyJodHRwczovL2luZGV4LmRvY2tlci5pby92MS8iOnsidXNlcm5hbWUiOiJ1c2VyIiwicGFzc3dvcmQiOiJwYXNzIiwiYXV0aCI6ImRYTmxjanB3WVhOeiJ9fX0=

---
# 3. kubernetes.io/tls (TLS certificate)
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi...
  tls.key: LS0tLS1CRUdJTi...

---
# 4. kubernetes.io/basic-auth
apiVersion: v1
kind: Secret
metadata:
  name: basic-auth
type: kubernetes.io/basic-auth
stringData:
  username: admin
  password: t0p-Secret
```

### 5.2 Secret Creation Commands

```bash
# Opaque Secret (literal)
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password=secret123 \
  -n production

# Create from file
kubectl create secret generic ssh-key \
  --from-file=ssh-privatekey=~/.ssh/id_rsa \
  --from-file=ssh-publickey=~/.ssh/id_rsa.pub

# Docker registry secret
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=myuser \
  --docker-password=mytoken \
  --docker-email=user@example.com

# TLS secret
kubectl create secret tls app-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem
```

### 5.3 Using Secrets

```yaml
# Use as environment variables
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secrets
spec:
  containers:
  - name: app
    image: myapp:latest
    env:
    # Use specific key only
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: username
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: password
    # Use entire Secret as env vars
    envFrom:
    - secretRef:
        name: app-secrets

---
# Mount as volume
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secret-volume
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true
    - name: tls-volume
      mountPath: /etc/tls
      readOnly: true
  volumes:
  - name: secret-volume
    secret:
      secretName: app-secrets
      # Mount specific keys only
      items:
      - key: api-key
        path: api-key.txt
        mode: 0400  # File permissions
  - name: tls-volume
    secret:
      secretName: tls-secret

---
# Image Pull Secret
apiVersion: v1
kind: Pod
metadata:
  name: private-image-pod
spec:
  containers:
  - name: app
    image: ghcr.io/myorg/private-app:latest
  imagePullSecrets:
  - name: regcred
```

### 5.4 Secret Security Hardening

```yaml
# Secret encryption config (kube-apiserver)
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>
      - identity: {}  # Fallback (unencrypted)

---
# Restrict Secret access with RBAC
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["app-secrets"]  # Specific Secret only
  verbs: ["get"]
```

### 5.5 External Secret Management Tools

```yaml
# External Secrets Operator example
# Fetch from AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: aws-secret
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: SecretStore
  target:
    name: db-credentials  # K8s Secret name to create
  data:
  - secretKey: username
    remoteRef:
      key: production/db-credentials
      property: username
  - secretKey: password
    remoteRef:
      key: production/db-credentials
      property: password

---
# Sealed Secrets (for GitOps)
# Encrypted with kubeseal
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: mysecret
  namespace: production
spec:
  encryptedData:
    password: AgBy8hCi...encrypted-data...
```

---

## 6. Pod Security

### 6.1 Pod Security Standards

```
┌─────────────────────────────────────────────────────────────┐
│              Pod Security Standards (PSS)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Privileged                                                 │
│  ├── Unrestricted                                          │
│  └── For system Pods                                       │
│                                                             │
│  Baseline                                                   │
│  ├── Prevents known privilege escalation                  │
│  ├── Forbids hostNetwork, hostPID                         │
│  └── Suitable for most workloads                          │
│                                                             │
│  Restricted                                                 │
│  ├── Strong security policy                               │
│  ├── Non-root execution required                          │
│  ├── Read-only root filesystem                            │
│  └── For security-sensitive workloads                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Pod Security Admission

```yaml
# Apply security level to namespace
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # enforce: deny violations
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    # audit: record in audit log
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest
    # warn: show warning message
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest

---
# baseline level namespace
apiVersion: v1
kind: Namespace
metadata:
  name: staging
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/warn: restricted
```

### 6.3 Security Context

```yaml
# secure-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  # Pod-level security context
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 3000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault

  containers:
  - name: app
    image: myapp:latest
    # Container-level security context
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
          - ALL
        # Add only necessary capabilities
        # add:
        #   - NET_BIND_SERVICE

    # Resource limits
    resources:
      limits:
        cpu: "500m"
        memory: "128Mi"
      requests:
        cpu: "250m"
        memory: "64Mi"

    # Temporary volumes (for read-only root when writes needed)
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/cache

  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir:
      sizeLimit: 100Mi
```

### 6.4 Advanced Security Configuration

```yaml
# highly-secure-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      # Don't mount ServiceAccount token
      automountServiceAccountToken: false

      # Pod security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534  # nobody
        runAsGroup: 65534
        fsGroup: 65534
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: app
        image: myapp:latest
        imagePullPolicy: Always

        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL

        # Ports
        ports:
        - containerPort: 8080
          protocol: TCP

        # Resource limits
        resources:
          limits:
            cpu: "1"
            memory: "512Mi"
          requests:
            cpu: "100m"
            memory: "128Mi"

        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: config
          mountPath: /etc/app
          readOnly: true

      volumes:
      - name: tmp
        emptyDir:
          medium: Memory
          sizeLimit: 64Mi
      - name: config
        configMap:
          name: app-config

      # Forbid host network/PID
      hostNetwork: false
      hostPID: false
      hostIPC: false

      # DNS policy
      dnsPolicy: ClusterFirst
```

### 6.5 Security Scanning

```bash
# Image vulnerability scanning (Trivy)
trivy image myapp:latest

# Cluster security scan (kubescape)
kubescape scan framework nsa --exclude-namespaces kube-system

# Pod security check (kube-bench)
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml
kubectl logs job/kube-bench

# OPA/Gatekeeper policy check
kubectl get constrainttemplates
kubectl get constraints
```

---

## 7. Practice Exercises

### Exercise 1: Development Team RBAC Configuration
```yaml
# Requirements:
# - Developers can manage Pods, Deployments, Services in development namespace
# - In production namespace, can only view Pods
# - No access to Secrets

# Write Role and RoleBinding
```

### Exercise 2: Microservices NetworkPolicy
```yaml
# Requirements:
# - Communication only: frontend -> api-gateway -> backend -> database
# - Allow monitoring namespace to access /metrics on all Pods
# - Only frontend accessible from outside

# Write NetworkPolicy
```

### Exercise 3: Secure Application Deployment
```yaml
# Requirements:
# - Run as non-root user
# - Read-only root filesystem
# - Drop all capabilities
# - Set resource limits
# - Mount Secrets as both env vars and volumes

# Write Deployment
```

### Exercise 4: Security Audit
```bash
# Check the following:
# 1. Find privileged Pods in cluster
# 2. Find Pods using default ServiceAccount
# 3. Find Pods with Secrets exposed as env vars
# 4. Find namespaces without NetworkPolicy

# Write commands
```

---

## Next Steps

- [08_Kubernetes_Advanced](08_Kubernetes_Advanced.md) - Ingress, StatefulSet, PV/PVC
- [09_Helm_Package_Management](09_Helm_Package_Management.md) - Helm chart management
- [10_CI_CD_Pipelines](10_CI_CD_Pipelines.md) - Automated deployment

## References

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [RBAC Documentation](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)

---

[← Previous: Docker Compose](06_Docker_Compose.md) | [Next: Kubernetes Advanced →](08_Kubernetes_Advanced.md) | [Table of Contents](00_Overview.md)
