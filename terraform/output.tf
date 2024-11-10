################################################################################
# Connection Detail Output
################################################################################

output "execute_this_to_access_the_test_dev_host" {
  value = "sudo ssh ubuntu@${aws_eip.eip_test.public_ip} -i cert.pem"
}

output "execute_this_to_access_the_deh_ckh_host" {
  value = "sudo ssh ubuntu@${aws_eip.eip_deh_ckh_dev.public_ip} -i cert.pem"
}

output "execute_this_to_access_the_deh_gb_host" {
  value = "sudo ssh ubuntu@${aws_eip.eip_deh_gb_dev.public_ip} -i cert.pem"
}

output "execute_this_to_access_the_deh_ckh_premium_host" {
  value = "sudo ssh ubuntu@${aws_eip.eip_deh_ckh_prem.public_ip} -i cert.pem"
}

output "execute_this_to_access_the_deh_gb_premium_host" {
  value = "sudo ssh ubuntu@${aws_eip.eip_deh_gb_prem.public_ip} -i cert.pem"
}