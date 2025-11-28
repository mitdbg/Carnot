variable "auth0_custom_domain" {
  description = "The custom domain you want to use with Auth0"
  type        = string
}

variable "auth0_cname_target" {
  description = "The CNAME value that Auth0 requires"
  type        = string
}

variable "hosted_zone_id" {
  description = "The Route 53 Hosted Zone ID for the domain."
  type        = string
  default     = "Z0371749174EVZHL80QS0"
}

variable "aws_region" {
  description = "AWS region for global resources"
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "The VPC ID where the EC2 instance will be launched."
  type        = string
  default     = "vpc-d27da3b7"
}

variable "deployer_public_key" {
  description = "The public SSH key for the deployer."
  type        = string
  default     = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKLX7sunnyaSl6i2JCO3fkSLU51lSwZShWCX8p5dbnGQ"
}

variable "acm_certificate_arn" {
  description = "The ARN of the ACM certificate for the ALB HTTPS listener."
  type        = string
  default     = "arn:aws:acm:us-east-1:422297141788:certificate/10be4bdb-461d-4dec-9ddf-501e2f157fab"
}

variable "public_subnet_ids" {
  description = "List of public subnet IDs for the ALB."
  type        = list(string)
  default     = ["subnet-27021a61", "subnet-09ffc721"]
}
