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

# -------------------------------
# S3 Bucket for Homepage
# -------------------------------
resource "aws_s3_bucket" "homepage" {
  bucket = "carnot-research-homepage"

  tags = {
    Name = "Carnot Homepage"
  }
}

resource "aws_s3_bucket_ownership_controls" "homepage_ownership" {
  bucket = aws_s3_bucket.homepage.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "public_bucket_access_block" {
  bucket = aws_s3_bucket.homepage.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_website_configuration" "homepage_website" {
  bucket = aws_s3_bucket.homepage.id

  index_document {
    suffix = "index.html"
  }
}

# The ALB needs permission to read from the S3 bucket via the S3 endpoint
resource "aws_s3_bucket_policy" "homepage_policy" {
  bucket = aws_s3_bucket.homepage.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowPublicRead"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.homepage.arn}/*"
      }
    ]
  })
  depends_on = [
    aws_s3_bucket_ownership_controls.homepage_ownership,
    aws_s3_bucket_public_access_block.public_bucket_access_block
  ]
}

# -------------------------------
# ALB Listener Rule for Root Domain (Homepage) - HTTPS
# -------------------------------
resource "aws_lb_listener_rule" "homepage_rule_https" {
  listener_arn = aws_lb_listener.https_listener.arn
  priority     = 1

  action {
    type = "redirect"
    redirect {
      host        = aws_s3_bucket_website_configuration.homepage_website.website_endpoint
      path        = "/#{path}"
      query       = "#{query}"
      protocol    = "HTTP"
      status_code = "HTTP_302"
    }
  }

  condition {
    host_header {
      values = ["carnot-research.org"]
    }
  }
}

# -------------------------------
# ALB Listener Rule for Root Domain (Homepage) - HTTP
# -------------------------------
resource "aws_lb_listener_rule" "homepage_rule_http" {
  listener_arn = aws_lb_listener.http_redirect.arn
  priority     = 1

  action {
    type = "redirect"
    redirect {
      host        = "carnot-research.org"
      path        = "/#{path}"
      query       = "#{query}"
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }

  condition {
    host_header {
      values = ["carnot-research.org"]
    }
  }
}

# -------------------------------
# Route53 Record for Root Domain
# -------------------------------
resource "aws_route53_record" "root_domain" {
  zone_id = var.hosted_zone_id
  name    = "carnot-research.org"
  type    = "A"

  alias {
    name                   = aws_lb.global_alb.dns_name
    zone_id                = aws_lb.global_alb.zone_id
    evaluate_target_health = true
  }
}