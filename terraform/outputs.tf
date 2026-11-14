output "instance_public_ip" {
  description = "Public IPv4 address of the Face Recognition EC2 instance"
  value       = aws_instance.app_server.public_ip
}

output "instance_id" {
  description = "EC2 Instance ID"
  value       = aws_instance.app_server.id
}

output "instance_public_dns" {
  description = "Public DNS of the EC2 instance"
  value       = aws_instance.app_server.public_dns
}
