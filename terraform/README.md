# EC2 Instance-Based Development

This folder provides Terraform modules to implement an EC2 development environment within an isolated "deh-vpc" Virtual Private Cloud.  Most configurations can be found in the [variables.tf](variables.tf) file.

The following infrastructure components are created:

* Virtual Private Cloud (VPC) to house development instance
* Internet Gateway and Subnets
* Elastic IP to provide allocated IP address (optional)
* Security Groups with Ingress rules for HTTP and SSH
* Private Key generation (cert.pem) for SSH connectivity
* EC2 Instance for remote development environment

## Pre-Requisites

It is expected that you have the following pre-requisites installed and functional.

* [Terraform client](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

It is also recommended that you have:

* VS Code to enable remote development.  Including the VS Code [SSH Remote Development Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).

## Executing Modules

Terraform modules can be run with the following commands:

`terraform plan` to confirm no issues within the module defintions.

`terraform apply` to execute modules and create cloud-based infrastructure

`terraform destroy` to remove cloud-based infrastructure.

## Login

* ssh command...
* git clone [https://github.com/rpc0/dehallucinator.git](https://github.com/rpc0/dehallucinator.git)
* cd dehallucinator
* docker compose build
* deh_measure build (utils/build...)
* download squad data (utils/dl...)
* docker exec -it ollama bash
* ollama pull llama3.1:8b-instruct-q3_K_L
* ollama pull mxbai-embed-large:latest

## VS Code

* Remote SSH connection
* Windows change permissions on cert.pem file (remove inheritance)

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all

## Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]

makes sure nvidia toolkit installed

sudo systemctl daemon-reload
sudo systemctl restart docker