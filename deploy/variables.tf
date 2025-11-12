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
