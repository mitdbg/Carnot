# NOTE: I needed to execute the following command once to initialize the DynamoDB table for state locking:
# aws dynamodb create-table \
#     --table-name carnot-terraform-locks \
#     --attribute-definitions AttributeName=LockID,AttributeType=S \
#     --key-schema AttributeName=LockID,KeyType=HASH \
#     --billing-mode PAY_PER_REQUEST \
#     --region us-east-1

terraform {
  backend "s3" {
    bucket         = "carnot-research"
    key            = "tf/global/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "carnot-terraform-locks"
    encrypt        = true
  }
}