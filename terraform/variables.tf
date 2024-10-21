# This variable defines the AWS Region.
variable "region" {
  description = "region to use for AWS resources"
  type        = string
  default     = "us-east-2"
}

variable "az" {
    description = "availability zone for AWS resources"
    type = string
    default = "us-east-2c"
}

variable "ec2_instance" {
  description = "EC2 instance to provision for dev"
  type = string
  default = "g4dn.2xlarge" # "c7i.4xlarge" # "t2.micro"
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