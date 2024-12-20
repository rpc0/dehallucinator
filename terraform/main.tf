################################################################################
# General
################################################################################

resource "aws_vpc" "default" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = {
    Name = "Deh VPC"
  }
}

resource "aws_internet_gateway" "default" {
  vpc_id = aws_vpc.default.id
  tags = {
    Name = "Deh Gateway"
  }
}

resource "aws_route" "default" {
  route_table_id         = aws_vpc.default.main_route_table_id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.default.id
}

resource "aws_subnet" "dev_host_subnet" {
  vpc_id                  = aws_vpc.default.id
  cidr_block              = var.cidr_blocks_dev_host[0]
  map_public_ip_on_launch = true
  availability_zone       = var.az
  tags = {
    Name = "Deh Subnet"
  }
}

################################################################################
# Security groups
################################################################################

resource "aws_security_group" "dev_host" {
  name   = "deh-dev-host"
  vpc_id = aws_vpc.default.id
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "tls_private_key" "private_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "private_key" {
  key_name   = var.global_prefix
  public_key = tls_private_key.private_key.public_key_openssh
}

resource "local_file" "private_key" {
  content  = tls_private_key.private_key.private_key_pem
  filename = "cert.pem"
}

resource "null_resource" "private_key_permissions" {
  depends_on = [local_file.private_key]
  provisioner "local-exec" {
    command     = "chmod 600 cert.pem"
    interpreter = ["bash", "-c"]
    on_failure  = continue
  }
}


################################################################################
# Client Machines - Test
################################################################################

resource "aws_instance" "test_dev_host" {
  ami                    = var.ami
  instance_type          = var.test_ec2_instance
  key_name               = aws_key_pair.private_key.key_name
  user_data              = templatefile("deh_ec2.tftpl", {})
  vpc_security_group_ids = [aws_security_group.dev_host.id]
  subnet_id              = aws_subnet.dev_host_subnet.id
  tags = {
    Name = "deh_test_ec2"
  } 
  root_block_device {
    volume_type = "gp2"
    volume_size = 25
  }
}

resource "aws_eip" "eip_test" {
  depends_on = [aws_internet_gateway.default, aws_instance.test_dev_host]
  domain     = "vpc"
  instance   = aws_instance.test_dev_host.id
}

################################################################################
# Client Machines - Dev
################################################################################

resource "aws_instance" "deh_ckh_host" {
  ami                    = var.ami
  instance_type          = var.dev_ec2_instance
  key_name               = aws_key_pair.private_key.key_name
  user_data              = templatefile("deh_ec2.tftpl", {})
  vpc_security_group_ids = [aws_security_group.dev_host.id]
  subnet_id              = aws_subnet.dev_host_subnet.id
  tags = {
    Name = "deh_ckh_dev"
  } 
  root_block_device {
    volume_type = "gp2"
    volume_size = 100
  }
}

resource "aws_eip" "eip_deh_ckh_dev" {
  depends_on = [aws_internet_gateway.default, aws_instance.deh_ckh_host]
  domain     = "vpc"
  instance   = aws_instance.deh_ckh_host.id
}

resource "aws_instance" "deh_gb_host" {
  ami                    = var.ami
  instance_type          = var.dev_ec2_instance
  key_name               = aws_key_pair.private_key.key_name
  user_data              = templatefile("deh_ec2.tftpl", {})
  vpc_security_group_ids = [aws_security_group.dev_host.id]
  subnet_id              = aws_subnet.dev_host_subnet.id
  tags = {
    Name = "deh_gb_dev"
  } 
  root_block_device {
    volume_type = "gp2"
    volume_size = 100
  }
}

resource "aws_eip" "eip_deh_gb_dev" {
  depends_on = [aws_internet_gateway.default, aws_instance.deh_gb_host]
  domain     = "vpc"
  instance   = aws_instance.deh_gb_host.id
}

################################################################################
# Client Machines - Premium
################################################################################

resource "aws_instance" "deh_ckh_prem_host" {
  ami                    = var.ami
  instance_type          = var.premium_ec2_instance
  key_name               = aws_key_pair.private_key.key_name
  user_data              = templatefile("deh_ec2.tftpl", {})
  vpc_security_group_ids = [aws_security_group.dev_host.id]
  subnet_id              = aws_subnet.dev_host_subnet.id
  tags = {
    Name = "deh_ckh_premium"
  } 
  root_block_device {
    volume_type = "gp2"
    volume_size = 100
  }
}

resource "aws_eip" "eip_deh_ckh_prem" {
  depends_on = [aws_internet_gateway.default, aws_instance.deh_ckh_prem_host]
  domain     = "vpc"
  instance   = aws_instance.deh_ckh_prem_host.id
}

resource "aws_instance" "deh_gb_prem_host" {
  ami                    = var.ami
  instance_type          = var.premium_ec2_instance
  key_name               = aws_key_pair.private_key.key_name
  user_data              = templatefile("deh_ec2.tftpl", {})
  vpc_security_group_ids = [aws_security_group.dev_host.id]
  subnet_id              = aws_subnet.dev_host_subnet.id
  tags = {
    Name = "deh_gb_premium"
  } 
  root_block_device {
    volume_type = "gp2"
    volume_size = 100
  }
}

resource "aws_eip" "eip_deh_gb_prem" {
  depends_on = [aws_internet_gateway.default, aws_instance.deh_gb_prem_host]
  domain     = "vpc"
  instance   = aws_instance.deh_gb_prem_host.id
}