drop database if exists slurm_acct_db;
create database slurm_acct_db;
drop user if exists 'slurm'@'%';
create user 'slurm'@'%';
set password for 'slurm'@'%' = password('some_pass');
grant usage on *.* to 'slurm'@'%';
grant all privileges on slurm_acct_db.* to 'slurm'@'%';
flush privileges;
exit
