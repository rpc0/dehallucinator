# DeHallucinator

As part of Harvard DCE Data Science Capstone (CSCI E-599a) this provides implementation of techniques to limit hallucination exposure to end-users as part of a RAG/LLM system.  Project includes development of 3 core modules:

* Measurement - provides a consistent approach to measuring RAG outcome (i.e. "answer") performance leveraging curated QA datasets.
* User Interface - provides capabilities to interact and refine with RAG chain processing to reduce hallucinated outputs.
* RAG API - provides implementation of advanced RAG approachs to provide guard-rails and mitigations to hallucinated outputs.

[![python](https://img.shields.io/badge/Python-3.9-3776AB?logo=Python&logoColor=white)](https://python.org/)

## Developer Setup

Project components are deployed via Docker containers with Docker-Compose orchestration.  High-level development architecture can be viewed [here](docs/docker_architecture.pptx) and additional details on specific Dockerized modules can be viewed [here](docs/docker_containers.md).  Module specific local development instructions can be found in READMEs within sub-project folders (e.g. [measurement](./measurement/), [rag_api](./rag_api/), and [ui](./ui/) respectively).

### IDE (Optional - VS Code)

It is recommended to use [VSCode](https://code.visualstudio.com/) as development IDE.  Specific Extensions/Configurations that are recommended include:

* Black Formatter (from Microsoft) - for automatic code formatting to PEP specifications
* Thunderbird Client - for REST API interactions
* JSON (from ZainChen) - for JSON file formatting
* Jupyter (from Microsoft) - for Jupyter Notebook support
* WSL (from Microsoft) - if using Windows WSL development environment
* Docker (from Microsoft) - for Docker file and resource management
* markdownlint - for Markdown (MD) file linting

### Windows Users (Recommended - WSL)

For Windows-based developers it is recommended that you [install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to provide a Linux-based (Ubuntu) tool chain environment.  It is possible to do project development directly in Windows environment given the use of Docker, but may pose some dependency conflict challenges.  Additionally majority of configuration examples will be provided assuming a Linux-based CLI.

### Docker installation

Depending on specific operating system installation of Docker may require different steps.  Developer may choose to install [Docker Engine (Docker CE)](https://docs.docker.com/engine/install/) w/ CLI support or full [Docker Desktop](https://docs.docker.com/desktop/).  Docker Desktop may be easier for developers less experienced with Docker to install and manage.

Once Docker is installed you should see Docker icon in taskbar (if using Desktop) or be able to successfully run commands like:

```bash
> sudo docker ps
```

As well as run test containers:

```bash
> sudo docker run hello-world
```

#### Enabling NVIDIA GPU Support for Docker

Given the use of LLMs via Ollama it is preferred to enable GPU support within your Docker containers.  Installation and configuration for GPU support via the NVIDIA container toolkit can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

If developer is unable to enable GPU support, Ollama docker-compose configuration can be adjusted to provide CPU-supported execution - though performance would be likley reduced.  An example of CPU configuration can be found [here](https://github.com/valiantlynx/ollama-docker/blob/main/docker-compose.yml) which basically entails removing the *deploy* section of the ollama service definition.  Comparison shown below:

![gpu-vs-cpu-comparison](docs/images/ollama-gpu-vs-cpu-config.png)

#### Docker-Compose Plugin

If you have installed Docker Desktop or docker-ce then Docker Compose should already be included in installation.  You can confirm via:

```bash
> sudo docker-compose version
```

If not available you may want to re-run [latest version installation](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) via:

```bash
>  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### BuildX Plugin

The [BuildX plugin](https://github.com/docker/buildx) brings additional Build functionality to the `docker build` command and is required for project Docker builds.  Plugin can be installed via below if not already installed:

```bash
>  sudo apt-get install docker-buildx-plugin
```

## DEH Environment Launch via Docker Compose

The DEH environment can be launched via Docker Compose via the following:

```bash
./dehallucinator> sudo docker-compose start
```

**Note:** Command run from root directory of the code-base - e.g. the 'dehallucinator' directory that contains the docker-compose.yml file

Upon successful execution of docker-compose command you should see status of component launches and output of container logs:

![docker-compose-start](./docs/images/docker-compose-start-up.png)

### Useful Commands

Connect to a specific Docker container:

```bash
> sudo docker exec -it [container_name] bash
```

Stop and start a specific service:

```bash
> sudo docker-compose stop [service name]
> sudo docker-compose start [service name]
```

Launch containers "detached" so that they run in background:

```bash
> sudo docker-compose up -d
```

Stream specific docker container logs:

```bash
> sudo docker logs --follow [container name]
```

View docker container resource utilization:

```bash
> sudo docker stats
```

## Download QA Data Corpus

The `data` folder is used to store specific document corpus files and is a local folder that is mounted to the `rag-api` docker container.  You can place whatever files you want in this container for RAG context processing (the folder is processed at the startup of the `rag-api` service).

The following steps will enable the developer to download the [SQuADv2.0](https://rajpurkar.github.io/SQuAD-explorer/) data-set in a standard format.

Change directory to measure folder:

```bash
> cd measurement
```

Create and activate a Python virtual environment:

```bash
> conda create -y --name=deh_measure_ python=3.9
> conda activate deh_measure
```

Install deh_measure packages:

```bash
> pip install -e .
```

Run SQUAD download utility (command line arguments can be included if desired but default to project compatible values):

```bash
> python src/dl/squad.py
```

Result of utility should be create of 3 folders in `data`:

![data-folder](./docs/images/data-folder.png)

* contexts - the document corpus of "raw information" that is used by the RAG chain.
* qa_dl_cache - a cache of the SQUAD QA dataset to prevent need to redownload in future processing.
* qas - a TSV file of questions and correct "ground truth" answer

The current SQUAD data-set includes over a thousand contexts and tens of thousands of question/answers so you may want to delete/limit for testing purposes.

## Developer Best Practices

* Do not directly commit code changes to `main` branch.  Use branches for development and create a PR (Pull Request) for merging.
* Scope branches for a specific change.  A branch should be short-lived to accomplish a particular goal.  Long-lived branches are often hard to merge into main due to code-drift.
* Do not merge code to main that fails Code Quality checks or other CI/CD assessments.  Take the time to clean up code formatting, etc. to keep code-base manageable.
* Add comments, doc-strings and other explanations to help others understand and build off of your code contributions.
