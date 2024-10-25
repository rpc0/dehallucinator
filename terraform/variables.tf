# This variable defines the AWS Region.
variable "region" {
  description = "region to use for AWS resources"
  type        = string
  default     = "us-east-2"
}

variable "az" {
    description = "availability zone for AWS resources"
    type = string
    default = "us-east-2a"
}

variable "ami" {
  description = "Ubuntu 24.04 LTS x86/HVM"
  type = string
  default = "ami-0ea3c35c5c3284d82"
}

variable "test_ec2_instance" {
  description = "Micro EC2 instance to provision for testing"
  type = string
  default = "t2.micro"
}

variable "premium_ec2_instance" {
  description = "P3 EC2 instance to provision for premium dev"
  type = string
  default = "p3.2xlarge"
}

variable "dev_ec2_instance" {
  description = "G EC2 instance to provision for dev"
  type = string
  default = "g4dn.2xlarge" # "c7i.4xlarge"
}

variable "global_prefix" {
  type    = string
  default = "deh"
}

variable "private_cidr_blocks" {
  type = list(string)
  default = [
    "10.0.1.0/24",
    "10.0.2.0/24",
    "10.0.3.0/24",
  ]
}

variable "cidr_blocks_dev_host" {
  type = list(string)
  default = ["10.0.4.0/24"]
}