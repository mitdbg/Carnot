output "alb_arn" {
  value = aws_lb.global_alb.arn
}

output "alb_sg_id" {
  value = aws_security_group.alb_sg.id
}

output "alb_dns_name" {
  value = aws_lb.global_alb.dns_name
}

output "alb_zone_id" {
  value = aws_lb.global_alb.zone_id
}

output "global_key_name" {
  value = aws_key_pair.deployer_key.key_name
}
