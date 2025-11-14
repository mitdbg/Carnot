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

# create security groups for ALB and EC2 instance
resource "aws_security_group" "alb_sg" {
  name   = "alb_sg"
  vpc_id = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
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

resource "aws_security_group" "allow_app" {
  name   = "allow_app"
  vpc_id = var.vpc_id

  ingress {
    description = "Allow ALB access to web service"
    from_port   = 5173
    to_port     = 5173
    protocol    = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Allows SSH from anywhere. Restrict for production.
    description = "Allow SSH from anywhere"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # Allow all protocols
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name = "allow_ssh_security_group"
  }
}

# create EBS volume for persistent storage
resource "aws_ebs_volume" "app_pg_data_volume" {
  availability_zone = var.availability_zone
  size              = var.ebs_volume_size
  type              = "gp2"

  tags = {
    Name = "carnot_app_pg_data_volume"
  }
}

resource "aws_volume_attachment" "ebs_volume_attachment" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.app_pg_data_volume.id
  instance_id = aws_instance.app_server.id
}

# create ALB and attach HTTPS listener using ACM certificate
resource "aws_lb" "app_alb" {
  name               = "app-alb"
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = var.public_subnet_ids  # list of 2+ public subnets
}

resource "aws_lb_listener" "https_listener" {
  load_balancer_arn = aws_lb.app_alb.arn
  port              = 443
  protocol          = "HTTPS"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app_tg.arn
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.app_alb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# create target group for EC2 instance
resource "aws_lb_target_group" "app_tg" {
  name     = "app-tg"
  port     = 5173
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {
    path = "/"
    port = "5173"
  }
}

resource "aws_lb_target_group_attachment" "app_attachment" {
  target_group_arn = aws_lb_target_group.app_tg.arn
  target_id        = aws_instance.app_server.id
  port             = 5173
}

# create A record in Route53 to point to ALB
resource "aws_route53_record" "app_dns" {
  zone_id = var.hosted_zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_lb.app_alb.dns_name
    zone_id                = aws_lb.app_alb.zone_id
    evaluate_target_health = true
  }
}

# EC2 Instance for the web application 
resource "aws_instance" "app_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  iam_instance_profile   = aws_iam_instance_profile.carnot_web_app_profile.name
  key_name               = aws_key_pair.deployer_key.key_name
  vpc_security_group_ids = [
    aws_security_group.allow_ssh.id,
    aws_security_group.allow_app.id
  ]

  # mount the EBS volume at /mnt/pg-data and install docker compose
  user_data = <<-EOF
    #!/bin/bash
    # Mount EBS volume
    sudo mkfs -t ext4 /dev/xvdf # Format the volume (adjust filesystem type if needed)
    sudo mkdir -p /mnt/pg-data
    sudo mount /dev/xvdf /mnt/pg-data
    echo "/dev/xvdf /mnt/pg-data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab # Add to fstab for persistent mounts
    sudo mkdir -p /mnt/pg-data/data
    sudo chown -R 999:999 /mnt/pg-data # Change ownership to postgres user (UID 999)

    # Install Docker
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker ubuntu # Add the 'ubuntu' user to the docker group
  EOF

  tags = {
    Name = var.instance_name
  }
}
