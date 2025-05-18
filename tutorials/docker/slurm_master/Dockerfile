FROM kevinsawade/slurm-base

#RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
#RUN apt-get update && apt-get install -y \
#    wget \

#####################
# SLURM
#####################
# set up slurmctld
# RUN mkdir /var/spool/slurmctld
# RUN chown slurm:slurm /var/spool/slurmctld
# RUN chmod 755 /var/spool/slurmctld
# RUN mkdir -p /var/log/slurm
# RUN touch /var/log/slurmctld.log
# RUN chown slurm: /var/log/slurmctld.log
# RUN touch /var/log/slurm_jobacct.log /var/log/slurm_jobcomp.log
# RUN chown slurm: /var/log/slurm*
# RUN chown slurm: /var/log/slurm/
# RUN chown slurm:slurm /etc/slurm
COPY slurm_base/slurm.conf /usr/etc/slurm.conf
COPY slurm_base/cgroup.conf /usr/etc/cgroup.conf
COPY slurm_base/cgroup_allowed_devices_file.conf /usr/etc/cgroup_allowed_devices_file.conf

# 6819
# run the entrypoint
COPY slurm_master/slurm_master_entrypoint.sh slurm_master_entrypoint.sh
RUN chmod a+wrx slurm_master_entrypoint.sh
EXPOSE 6817 6818 6819
CMD ["/slurm_master_entrypoint.sh"]
