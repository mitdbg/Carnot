variable "instance_name" {
  description = "Value of the EC2 instance's Name tag."
  type        = string
  default     = "carnot-web-app"
}

variable "instance_type" {
  description = "The EC2 instance's type."
  type        = string
  default     = "m5.xlarge"
}

variable "subnet_id" {
  description = "The subnet ID where the EC2 instance will be launched."
  type        = string
  default     = "subnet-27021a61"
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

# NOTE: this must match the AZ of the subnet where the instance is launched
variable "availability_zone" {
  description = "The availability zone for the EBS volume."
  type        = string
  default     = "us-east-1a"
}

variable "ebs_volume_size" {
  description = "The size of the EBS volume in GB."
  type        = number
  default     = 10
}

variable "public_subnet_ids" {
  description = "List of public subnet IDs for the ALB."
  type        = list(string)
  default     = ["subnet-27021a61", "subnet-09ffc721"]
}

variable "acm_certificate_arn" {
  description = "The ARN of the ACM certificate for the ALB HTTPS listener."
  type        = string
  default     = "arn:aws:acm:us-east-1:422297141788:certificate/10be4bdb-461d-4dec-9ddf-501e2f157fab"
}

variable "hosted_zone_id" {
  description = "The Route 53 Hosted Zone ID for the domain."
  type        = string
  default     = "Z0371749174EVZHL80QS0"
}

variable "domain_name" {
  description = "The domain name for the application."
  type        = string
  default     = "carnot-research.org"
}