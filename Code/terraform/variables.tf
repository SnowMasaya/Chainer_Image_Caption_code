# Configure the AWS Provider
variable "access_key" {}
variable "secret_key" {}

variable "aws_region" {
  description = "The AWS region to create things in."
  default = "us-east-1"
}

# Amazon Linux AMI 2015.09.2 x86_64 Graphics HVM EBS
variable "aws_amis" {
  default = {
    "us-east-1" = "ami-ebcec381"
  }
}

variable "key_name" {
  default = "chainer_image_caption"
  description = "Name of the SSH keypair to use in AWS."
}
