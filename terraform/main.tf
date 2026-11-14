# 1. Query the latest official Amazon Linux 2023 AMI
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# 2. Query the existing Default VPC
data "aws_vpc" "default" {
  default = true
}

# 3. Query existing subnets in the Default VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# 4. Create exactly one EC2 instance using existing resources
resource "aws_instance" "app_server" {
  ami           = data.aws_ami.al2023.id
  instance_type = var.instance_type

  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [var.security_group_id]
  key_name                    = var.key_name
  iam_instance_profile        = var.iam_instance_profile
  associate_public_ip_address = true

  # IMDSv2 only (session tokens required). Hop limit 2 lets the app running inside the Docker
  # container reach the instance role; with hop limit 1 the container could not get S3/ECR credentials.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  root_block_device {
    volume_size           = 15
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = {
    Name = var.instance_name
  }

  lifecycle {
    # The AMI data source above always returns the newest Amazon Linux 2023 image. Without this,
    # a later "terraform plan" could propose REPLACING the running server just because Amazon
    # published a new AMI. To adopt a new AMI deliberately: terraform apply -replace=aws_instance.app_server
    ignore_changes = [ami]
  }
}
