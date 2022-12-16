# liveProject-anomalyDetection

## Part 1

### 1.1 Objective
Define a Docker image that runs a Jupyter server. Next, create a Jupyter notebook by accessing Jupyter from the host machine. In Milestones 2 and 3, we will use this Jupyter notebook to explore the dataset and train the anomaly detection model.

### 1.2 Importance
- Running processes in Docker containers is a common practice in the industry for many reasons, including the portability and ease of sharing images, its ability to run almost everywhere, the isolation it provides, and because you can avoid installing dependencies on your machine.
- By defining, building, and running a Docker image, we’ll start to familiarize ourselves with some of the terms we will use in the project: for example, image and container.
- We will use the same Jupyter notebook we just created to analyze our dataset and train the model.
- We will create another Dockerfile to run the web service using a similar approach in a later milestone.

### 1.3 Workflow

1. Create a new directory and name it “liveProject.” This location is the working directory and is where we will develop our project.
2. Within the directory, create a second directory and name it “jupyter.” Then, in the Jupyter directory, create a Dockerfile. This Dockerfile represents the image we will use to run Jupyter.
3. In the Dockerfile, define as base a Python image. A smaller one like `python:3.7-slim` should suffice. Then, use the `RUN` instruction with the command `mkdir src` to create a `/src` directory inside the container. Use the `WORKDIR` instruction to set `/src` as the working directory.
4. In the Dockerfile, add another `RUN` that installs Jupyter using pip, which is the Python package installer.
5. Next, use the `CMD` instruction to specify the command that starts the Jupyter server.
    - Because we are inside a Docker container and we want to access Jupyter from the outside, we need to use the following options while starting the Jupyter server: `no-browser` and `allow-root`. Besides this, I’d recommend using the `ip` and `port` options to explicitly define the IP and port where the server will run. You could use the IP address `0.0.0.0`.
6. After defining the Dockerfile, on the terminal, use Docker’s `build` command to build the image using the name `{your_name}/lp-jupyter`: for example, `juan/lp-jupyter`.
7. Run the Docker image using the publish or expose option (`-p`) to forward the port you used in the second `CMD` command to the local machine. The command that runs the Docker image is named `run`. This command will run the Docker image and start the Jupyter server. After running it, you will see on screen the address where Jupyter is running. Click on it to access it.
8. From Jupyter, create a new notebook and print “hello, world.”
