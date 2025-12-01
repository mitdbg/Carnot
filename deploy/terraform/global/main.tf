provider "aws" {
  region = var.aws_region
}

# -------------------------------
# Global SSH key
# -------------------------------
resource "aws_key_pair" "deployer_key" {
  key_name   = "carnot-deployer-key"
  public_key = var.deployer_public_key
}

# -------------------------------
# Security Group for ALB
# -------------------------------
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

# -------------------------------
# ALB (shared across all envs)
# -------------------------------
resource "aws_lb" "global_alb" {
  name               = "carnot-global-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = var.public_subnet_ids

  tags = {
    Name = "carnot-global-alb"
  }
}

# -------------------------------
# ALB Listeners
# -------------------------------
resource "aws_lb_listener" "https_listener" {
  load_balancer_arn = aws_lb.global_alb.arn
  port              = 443
  protocol          = "HTTPS"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "Not Found"
      status_code  = "404"
    }
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.global_alb.arn
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

# ---------------------------------------------------------
# Route53 Record for Auth0 Custom Domain
# ---------------------------------------------------------
resource "aws_route53_record" "auth0_custom_domain" {
  zone_id = var.hosted_zone_id
  name    = var.auth0_custom_domain
  type    = "CNAME"
  ttl     = 300

  records = [var.auth0_cname_target]
}
