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

variable "public_subnet_ids" {
  description = "List of public subnet IDs for the ALB."
  type        = list(string)
  default     = ["subnet-27021a61", "subnet-09ffc721"]
}
