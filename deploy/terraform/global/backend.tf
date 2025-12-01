terraform {
  backend "s3" {
    bucket       = "carnot-research"
    key          = "tf/global/terraform.tfstate"
    region       = "us-east-1"
    use_lockfile = true
    encrypt      = true
  }
}