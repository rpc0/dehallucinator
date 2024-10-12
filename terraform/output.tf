################################################################################
# Diagnostic Output
################################################################################

output "execute_this_to_access_the_dev_host" {
  value = "sudo ssh ubuntu@${aws_instance.dev_host.public_ip} -i cert.pem"
}