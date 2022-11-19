# Introduction to kubernetes

## objectives

* Use kubectl CLI
* Create a kubernetes pod
* Create a kubernetes deployment
* Create a Replicaset that maintains a set number of replicas
* Witness kubernetes load balancing in action

## Use kubectl CLI

* At first we need to check whether it is there or not. We can see the version. ``kubectl version``
* Then we need to clone the repo. If already done in [last lab](https://hasangoni.github.io/2022/08/20/app_with_docker.html), then we don't need to clone again.
* go to desired directory by `cd /CC201/labs/2_IntroKubernetes/`
* Kubernetes namespace to virtualize a cluster. Kubectl already set to target that cluster and namespace
* `kubectl` requires configuration so that it targets the appropriate cluster. At first get cluster informatinb by running following command.
  ```bash
  kubectl config get-clusters
  ```
* `kubectl` context is a group of access parameters including cluster, a user, and a namespace. Can be found by
  ```bash
  kubectl config get-contexts
  ```
* get all pods information in users namespace.
  ```bash
  kubectl get pods
  ```

## Creating a pod

* Right now we don't have any pods. So we will create one. There are two ways to create a pod
  
  1. Imperative command
  2. Imperative object configuration
  3. Declarative command

### Creating pod with imperative command

* like earlier first we will build the docker image and push it to registry. The docker file content is the following.

  ```bash
  FROM node:9.4.0-alpine
  COPY app.js .
  COPY package.json .
  RUN npm install &&\
    apk update &&\
    apk upgrade
  EXPOSE  8080
  CMD node app.js
  ```
* Creating new environment variable to reuse later``export MY_NAMESPACE=sn-labs-$USERNAME``
* building and pushing can be done with the following command `docker build -t us.icr.io/$MY_NAMESPACE/hello-world:1 . && docker push us.icr.io/$MY_NAMESPACE/hello-world:1`
* Now run hello world by running following command ``kubectl run hello-world --image us.icr.io/$MY_NAMESPACE/hello-world:1 --overrides='{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"icr"}]}}}}'``. 
* The --overrides option here enables us to specify the needed credentials to pull this image from IBM Cloud Container Registry. Note that this is an imperative command, as we told Kubernetes explicitly what to do: run ``hello world``
* This is the imperative way creating pod, as we are telling everything
* Let see whether a pod is crated. ``kubectl get pods``
* We can see a pod is already created and a autogenerated name is also given. To get more information, we can use the wide option. ``kubectl get pods -o wide``
* We can get more information about the speicific pod. ``kubectl describe pod hello-world``
  > Note: The output shows the pod parameters like Namespace, Pod Name, IP address, the time when the pod started running and also the container parameters like container ID, image name & ID, running status and the memory/CPU limits.
* Now we will delete the pod. ``kubectl delete pod hello-world``
* let's check whether the pod is deleted. ``kubectl get pods``

### Creating pod with imperative object configuration

* Imperative object configuration can be done using a yaml file. We have a hello-world-create.yaml file in the repo. We can see the content of the file is following. 
```bash
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
spec:
  containers:
  - name: hello-world
    image: us.icr.io/<my_namespace>/hello-world:1
    ports:
    - containerPort: 8080
  imagePullSecrets:
  - name: icr
```
* In this file we need to change the <my_namespace>variable. we can use $MY_NAMESPACE environment variable, which we created earlier.
* Now to create the pod, we need to run the following command. `kubectl create -f hello-world-create.yaml`
* Now again we can check whether pod is available, `kubectl get pods`
* After that we again need to delete the pods. `kubectl delete pod hello-world`

## Creating a pod using declarative command(deployment, replica)

* Imperative way is very easy but productive environment it is good to use declarative way to create a pod
* In this repo, there a file with following content in it

  ``` bash
  apiVersion: apps/v1
kind: Deployment
metadata:
  generation: 1
  labels:
    run: hello-world
  name: hello-world
spec:
  replicas: 3
  selector:
    matchLabels:
      run: hello-world
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
    type: RollingUpdate
  template:
    metadata:
      labels:
        run: hello-world
    spec:
      containers:
      - image: us.icr.io/<my_namespace>/hello-world:1
        imagePullPolicy: Always
        name: hello-world
        ports:
        - containerPort: 8080
          protocol: TCP
        resources:
          limits:
            cpu: 2m
            memory: 30Mi
          requests:
            cpu: 1m
            memory: 10Mi   
      imagePullSecrets:
      - name: icr
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      securityContext: {}
      terminationGracePeriodSeconds: 30
  ```
* We are using deployment, namespace and image namae and replicas and cpu
* Now we can create a pod by `kubectl apply -f hello-world-build.yaml`
* Now we can get the deployment. `kubectl get deployments`

## Load balancing the application

1. ``kubectl expose deployment/hello-world`` - will expose the app. This command creates what is called a ClusterIP Service. This creates an IP address that accessible within the cluster.
2. List Services in order to see that this service was created. ``kubectl get services``
3. Open a new terminal window using Terminal > Split Terminal.
4. Since the cluster IP is not accessible outside of the cluster, we need to create a proxy. Note that this is not how you would make an application externally accessible in a production scenario. Run this command (kubectl proxy)in the new terminal window since your environment variables need to be accessible in the original window for subsequent commands.This command doesn't terminate until you terminate it. Keep it running so that you can continue to access your app.:w
5. In the original terminal window, ping the application to get a response.
```bash
curl -L localhost:8001/api/v1/namespaces/sn-labs-$USERNAME/services/hello-world/proxy
```
6. Run the command which runs a for loop ten times creating 10 different pods and note names for each new pod.
```bash
for i in `seq 10`; do curl -L localhost:8001/api/v1/namespaces/sn-labs-$USERNAME/services/hello-world/proxy; done
```
7. You should see more than one Pod name, and quite possibly all three Pod names, in the output. This is because Kubernetes load balances the requests across the three replicas, so each request could hit a different instance of our application.
8. Delete the Deployment and Service. This can be done in a single command by using slashes.``kubectl delete deployment/hello-world service/hello-world``
9. Return to the terminal window running the proxy command and kill it using Ctrl+C.

## Cheat sheet
Command Description

* kubectl apply ->Apply a configuration to a resource.
* kubectl config get-clusters >Displays clusters defined in the kubeconfig.
* kubectl config get-contexts >Displays the current context.
* kubectl create ->Create a resource.
* kubectl delete ->Deletes resources.
* kubectl describe -> Shows details of a resource or group of resources.
* kubectl expose ->Expose a resource to the internet as a new Kubernetes * service.
* kubectl get ->Displays resources.
* kubectl get pods >Lists all the Pods in the namespace.
* kubectl proxy >Creates a proxy server between a localhost and the Kubernetes API
server.
* kubectl rollout >Manage the rollout of a resource.
* kubectl run >Creates and runs a particular image in a pod.
* kubectl scale> Set a new size for a deployment.
* kubectl set >Configure application resources.
* kubectl version >Prints the client and server version information.