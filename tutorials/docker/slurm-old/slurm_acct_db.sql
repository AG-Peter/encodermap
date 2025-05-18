create database slurm_acct_db;
create user 'slurm'@'%';
set password for 'slurm'@'%' = password('some_pass');
grant usage on *.* to 'slurm'@'%';
grant all privileges on slurm_acct_db.* to 'slurm'@'%';
flush privileges;
