variable "instance_name" {
  description = "Value of the EC2 instance's Name tag."
  type        = string
  default     = "carnot-web-app"
}

variable "instance_type" {
  description = "The EC2 instance's type."
  type        = string
  default     = "t2.micro"
}

variable "subnet_id" {
  description = "The subnet ID where the EC2 instance will be launched."
  type        = string
  default     = "subnet-27021a61"
}

variable "deployer_public_key" {
  description = "The public SSH key for the deployer."
  type        = string
  default     = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKLX7sunnyaSl6i2JCO3fkSLU51lSwZShWCX8p5dbnGQ"
  
}
