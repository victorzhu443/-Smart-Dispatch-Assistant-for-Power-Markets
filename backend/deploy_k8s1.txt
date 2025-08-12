@echo off
REM deploy_k8s.bat - Phase 5.3: Windows Deployment Script

echo 🚀 Phase 5.3: Deploy Smart Dispatch System to Local Kubernetes

REM Check if minikube is installed
minikube version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ minikube is not installed
    echo Please install minikube: https://minikube.sigs.k8s.io/docs/start/
    pause
    exit /b 1
)

REM Check if kubectl is installed
kubectl version --client >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ kubectl is not installed
    echo Please install kubectl: https://kubernetes.io/docs/tasks/tools/
    pause
    exit /b 1
)

echo ✅ Prerequisites checked

REM Start minikube
echo 🔄 Starting minikube...
minikube start --driver=docker --memory=4096 --cpus=2
if %errorlevel% neq 0 (
    echo ❌ Failed to start minikube
    pause
    exit /b 1
)

REM Enable ingress
echo 🔄 Enabling ingress...
minikube addons enable ingress

REM Configure docker environment
echo 🔄 Configuring Docker environment...
for /f "tokens=*" %%i in ('minikube docker-env --shell cmd') do %%i

REM Create namespace
echo 🔄 Creating namespace...
echo apiVersion: v1 > k8s-namespace.yaml
echo kind: Namespace >> k8s-namespace.yaml
echo metadata: >> k8s-namespace.yaml
echo   name: smart-dispatch >> k8s-namespace.yaml

kubectl apply -f k8s-namespace.yaml

REM Create simple Dockerfile for forecast
echo 🔄 Building forecast API...
echo FROM python:3.9-slim > Dockerfile.forecast
echo WORKDIR /app >> Dockerfile.forecast
echo RUN apt-get update ^&^& apt-get install -y curl ^&^& rm -rf /var/lib/apt/lists/* >> Dockerfile.forecast
echo COPY requirements_minimal.txt ./requirements.txt >> Dockerfile.forecast
echo RUN pip install --no-cache-dir -r requirements.txt >> Dockerfile.forecast
echo COPY phase_5_1_forecast_api.py . >> Dockerfile.forecast
echo COPY market_data.db . >> Dockerfile.forecast
echo EXPOSE 5001 >> Dockerfile.forecast
echo CMD ["python", "phase_5_1_forecast_api.py"] >> Dockerfile.forecast

docker build -f Dockerfile.forecast -t smart-dispatch-forecast:latest .

REM Create simple Dockerfile for query
echo 🔄 Building query API...
echo FROM python:3.9-slim > Dockerfile.query
echo WORKDIR /app >> Dockerfile.query
echo RUN apt-get update ^&^& apt-get install -y curl ^&^& rm -rf /var/lib/apt/lists/* >> Dockerfile.query
echo COPY requirements_minimal.txt ./requirements.txt >> Dockerfile.query
echo RUN pip install --no-cache-dir -r requirements.txt >> Dockerfile.query
echo COPY phase_5_2_minimal.py . >> Dockerfile.query
echo COPY market_embeddings.json . >> Dockerfile.query
echo COPY gpt2_dispatch_model/ ./gpt2_dispatch_model/ >> Dockerfile.query
echo COPY market_data.db . >> Dockerfile.query
echo EXPOSE 5002 >> Dockerfile.query
echo CMD ["python", "phase_5_2_minimal.py"] >> Dockerfile.query

docker build -f Dockerfile.query -t smart-dispatch-query:latest .

REM Create simple deployment manifests
echo 🔄 Creating deployments...

REM Forecast deployment
echo apiVersion: apps/v1 > k8s-forecast.yaml
echo kind: Deployment >> k8s-forecast.yaml
echo metadata: >> k8s-forecast.yaml
echo   name: forecast-api >> k8s-forecast.yaml
echo   namespace: smart-dispatch >> k8s-forecast.yaml
echo spec: >> k8s-forecast.yaml
echo   replicas: 1 >> k8s-forecast.yaml
echo   selector: >> k8s-forecast.yaml
echo     matchLabels: >> k8s-forecast.yaml
echo       app: forecast-api >> k8s-forecast.yaml
echo   template: >> k8s-forecast.yaml
echo     metadata: >> k8s-forecast.yaml
echo       labels: >> k8s-forecast.yaml
echo         app: forecast-api >> k8s-forecast.yaml
echo     spec: >> k8s-forecast.yaml
echo       containers: >> k8s-forecast.yaml
echo       - name: forecast-api >> k8s-forecast.yaml
echo         image: smart-dispatch-forecast:latest >> k8s-forecast.yaml
echo         imagePullPolicy: Never >> k8s-forecast.yaml
echo         ports: >> k8s-forecast.yaml
echo         - containerPort: 5001 >> k8s-forecast.yaml
echo --- >> k8s-forecast.yaml
echo apiVersion: v1 >> k8s-forecast.yaml
echo kind: Service >> k8s-forecast.yaml
echo metadata: >> k8s-forecast.yaml
echo   name: forecast-service >> k8s-forecast.yaml
echo   namespace: smart-dispatch >> k8s-forecast.yaml
echo spec: >> k8s-forecast.yaml
echo   selector: >> k8s-forecast.yaml
echo     app: forecast-api >> k8s-forecast.yaml
echo   ports: >> k8s-forecast.yaml
echo   - port: 80 >> k8s-forecast.yaml
echo     targetPort: 5001 >> k8s-forecast.yaml
echo   type: ClusterIP >> k8s-forecast.yaml

kubectl apply -f k8s-forecast.yaml

REM Query deployment
echo apiVersion: apps/v1 > k8s-query.yaml
echo kind: Deployment >> k8s-query.yaml
echo metadata: >> k8s-query.yaml
echo   name: query-api >> k8s-query.yaml
echo   namespace: smart-dispatch >> k8s-query.yaml
echo spec: >> k8s-query.yaml
echo   replicas: 1 >> k8s-query.yaml
echo   selector: >> k8s-query.yaml
echo     matchLabels: >> k8s-query.yaml
echo       app: query-api >> k8s-query.yaml
echo   template: >> k8s-query.yaml
echo     metadata: >> k8s-query.yaml
echo       labels: >> k8s-query.yaml
echo         app: query-api >> k8s-query.yaml
echo     spec: >> k8s-query.yaml
echo       containers: >> k8s-query.yaml
echo       - name: query-api >> k8s-query.yaml
echo         image: smart-dispatch-query:latest >> k8s-query.yaml
echo         imagePullPolicy: Never >> k8s-query.yaml
echo         ports: >> k8s-query.yaml
echo         - containerPort: 5002 >> k8s-query.yaml
echo --- >> k8s-query.yaml
echo apiVersion: v1 >> k8s-query.yaml
echo kind: Service >> k8s-query.yaml
echo metadata: >> k8s-query.yaml
echo   name: query-service >> k8s-query.yaml
echo   namespace: smart-dispatch >> k8s-query.yaml
echo spec: >> k8s-query.yaml
echo   selector: >> k8s-query.yaml
echo     app: query-api >> k8s-query.yaml
echo   ports: >> k8s-query.yaml
echo   - port: 80 >> k8s-query.yaml
echo     targetPort: 5002 >> k8s-query.yaml
echo   type: ClusterIP >> k8s-query.yaml

kubectl apply -f k8s-query.yaml

REM Create ingress
echo apiVersion: networking.k8s.io/v1 > k8s-ingress.yaml
echo kind: Ingress >> k8s-ingress.yaml
echo metadata: >> k8s-ingress.yaml
echo   name: smart-dispatch-ingress >> k8s-ingress.yaml
echo   namespace: smart-dispatch >> k8s-ingress.yaml
echo spec: >> k8s-ingress.yaml
echo   ingressClassName: nginx >> k8s-ingress.yaml
echo   rules: >> k8s-ingress.yaml
echo   - host: localhost >> k8s-ingress.yaml
echo     http: >> k8s-ingress.yaml
echo       paths: >> k8s-ingress.yaml
echo       - path: /forecast >> k8s-ingress.yaml
echo         pathType: Prefix >> k8s-ingress.yaml
echo         backend: >> k8s-ingress.yaml
echo           service: >> k8s-ingress.yaml
echo             name: forecast-service >> k8s-ingress.yaml
echo             port: >> k8s-ingress.yaml
echo               number: 80 >> k8s-ingress.yaml
echo       - path: /query >> k8s-ingress.yaml
echo         pathType: Prefix >> k8s-ingress.yaml
echo         backend: >> k8s-ingress.yaml
echo           service: >> k8s-ingress.yaml
echo             name: query-service >> k8s-ingress.yaml
echo             port: >> k8s-ingress.yaml
echo               number: 80 >> k8s-ingress.yaml
echo       - path: /chat >> k8s-ingress.yaml
echo         pathType: Prefix >> k8s-ingress.yaml
echo         backend: >> k8s-ingress.yaml
echo           service: >> k8s-ingress.yaml
echo             name: query-service >> k8s-ingress.yaml
echo             por