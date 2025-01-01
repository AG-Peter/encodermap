FROM kevinsawade/slurm-base

#####################
# SOME ADDITIONAL STUFF
#####################

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sshpass

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

# run the entrypoint
WORKDIR /
RUN mkdir /work
COPY slurm_node/slurm_node_entrypoint.sh slurm_node_entrypoint.sh
COPY slurm_node/test_slurm_node.sh test_slurm_node.sh
COPY slurm_node/run_dask_tests_on_slurm.sh run_dask_tests_on_slurm.sh
RUN chmod a+wrx slurm_node_entrypoint.sh
RUN chmod a+wrx test_slurm_node.sh
RUN chmod a+wrx run_dask_tests_on_slurm.sh
EXPOSE 6817 6818 6819
CMD ["/slurm_node_entrypoint.sh"]
