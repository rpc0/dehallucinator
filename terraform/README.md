# EC2 Instance-Based Development

This folder provides Terraform modules to implement an EC2 development environment within an isolated "deh-vpc" Virtual Private Cloud.  Most configurations can be found in the [variables.tf](variables.tf) file.

The following infrastructure components are created:

* Virtual Private Cloud (VPC) to house development instance
* Internet Gateway and Subnets
* Elastic IP to provide allocated IP address (optional)
* Security Groups with Ingress rules for HTTP and SSH
* Private Key generation (cert.pem) for SSH connectivity
* Mid-Tier GPU EC2 (G4DN) Instance for remote development environment
* High-End GPU EC2 (P3) Instance for scaled measurement, experimentation

EC2 Instances are initialized via [start-up template](deh_ec2.tftpl) that facilitates installation of:

* Python3
* Docker and associated Compose and BuildX plugins
* Nvidia Drivers
* Nvidia Docker Toolkit
* Miniconda

## Pre-Requisites

It is expected that you have the following pre-requisites installed and functional.

* [Terraform client](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

## Executing Terraform Modules

Before running Terraform modules review associated [variables](variables.tf) and pre-connect to your AWS account via AWS CLI.

Terraform modules can be run with the following commands:

`terraform plan` to confirm no issues within the module defintions.

`terraform apply` to execute modules and create cloud-based infrastructure

`terraform destroy` to remove cloud-based infrastructure.

**Watch-out:** After `terraform apply` creation of EC2 Instances it may take a while for full installation of required libraries.  Installation progress can be monitored in EC2 Instance by review `more /deh_ec2_setup.txt` file.

**Watch-out:** Reboot of EC2 instance may be required after installation completion to make sure Nvidia drivers are activated.  Driver support can be checked via `nvidia-smi`.

## Post-Creation EC2 Instance Setup

### SSH Connection

After completion of Terraform EC2 instance creation the generated `cert.pem` file can be used as private key for SSH connection to instance via allocated EIP.  An output similiar to below will be provided from Terraform:

```bash
sudo ssh ubuntu@<generated_ip_address> -i cert.pem
```

### GitHub Clone

Once logged into EC2 instance you can clone the [DEH repository](https://github.com/rpc0/dehallucinator).  You could clone in read-only mode via:

```bash
git clone https://github.com/rpc0/dehallucinator.git
```

but it is recommended that you clone using SSH so that you can make changes directly from EC2 instance.  This will requiring [creating SSH key on EC2 instance and providing to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).  Cloning can then be done via:

```bash
git clone git@github.com:rpc0/dehallucinator.git
```

You may also have to set git username and email parameters to enable first commits:

```bash
git config --global user.name "User Name"
git config --global user.email "user@example.com"
```

### QA File Download

Follow typical approach to build deh_measures container and download QA file-set:

```bash
chmod +x ./measurement/utils/build_container.sh
chmod +x ./measurement/utils/download_squad_data.sh
./measurement/utils/build_container.sh
./measurement/utils/download_squad_data.sh
```

### Build & Launch Docker Infrastructure

```bash
docker compose -f docker-compose-gpu.yml build
docker compose -f docker-compose-gpu.yml up
```

**Note:** Make take a few minutes for initial container builds.

### Pull Ollama Models

Connect to the Ollama hosting container:

```bash
docker exec -it ollama bash
```

Pull required models (via command line within Ollama container):

```bash
ollama pull llama3.1:8b-instruct-q3_K_L
ollama pull mxbai-embed-large:latest
```

### API Confirmation

You should now be able to access your EC2 instance via typical api endpoints, e.g. `http://ec2-instance-ip/api/`.

## VS Code SSH Remote Development Setup

For a more productive experience, it is also recommended that you have:

* VS Code to enable remote development.  Including the VS Code [SSH Remote Development Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)

**Watch-out:** SSH connection from Windows may require changing permissions on file so that only logged in user can access (e.g. remove permission inheritance, etc.).

### Conda env creation

For developement it is required to create conda virtual environments per project.  As an example for execution of measurement notebooks:

```bash
conda create --name deh_measurement python=3.10
conda activate deh_measurement
cd ./measurement
pip install -e .
```

From within VS Code you should then make sure to select `deh_measurement` as the associated Python environment.
