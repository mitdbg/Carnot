provider "aws" {
  region = "us-east-1"
}

data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  owners = ["099720109477"] # Canonical
}

# create ssh key pair
resource "aws_key_pair" "deployer_key" {
  key_name   = "carnot-deployer-key"
  public_key = var.deployer_public_key
}

# create instance profile to attach IAM role to EC2 instance
data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "role" {
  name               = "CarnotWebAppRole"
  path               = "/"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

resource "aws_iam_instance_profile" "carnot_web_app_profile" {
  name = "CarnotWebAppInstanceProfile"
  role = aws_iam_role.role.name
}

# EC2 Instance for the web application 
resource "aws_instance" "app_server" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  subnet_id            = var.subnet_id
  iam_instance_profile = aws_iam_instance_profile.carnot_web_app_profile.name
  key_name             = aws_key_pair.deployer_key.key_name

  tags = {
    Name = var.instance_name
  }
}
