variable "aws_region" {
  description = "Target AWS Region"
  type        = string
  default     = "ap-south-1"
}

variable "instance_type" {
  description = "EC2 Instance Type"
  type        = string
  default     = "t3.small"
}

variable "key_name" {
  description = "Existing EC2 Key Pair Name in AWS"
  type        = string
  default     = "aws-demo-key-july"
}

variable "security_group_id" {
  description = "Existing Security Group ID to attach"
  type        = string
  default     = "sg-05b9179e2a38c02db"
}

variable "iam_instance_profile" {
  description = "Existing IAM Instance Profile Name to attach to EC2"
  type        = string
  default     = "FaceRecognitionEC2Role"
}

variable "instance_name" {
  description = "Value of the Name tag for the EC2 instance"
  type        = string
  default     = "Face-Recognition-Server"
}
