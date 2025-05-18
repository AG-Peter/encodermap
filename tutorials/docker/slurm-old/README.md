# slurm-master

access the mariadb server running in another container via:

```bash
mysql -u slurm -p some_pass -h slurm-db.local.dev
```

Compose with:

docker-compose up -d --build
