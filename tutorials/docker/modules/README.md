# Environment modules in docker

This docker-container is used as a base for a container with environment-modules installed in it. The `modules` command needs to be sourced first with

```bash
$ source /usr/share/Modules/init/profile.sh
```

Gromacs is then available as a module with:

```bash
module load gromacs/2023.1
```

## Build args

This container supports various build args:

- GOSU_VERSION (standard 1.11)
- ENVIRONMENT_MODULES_VERSION (standard 5.2.0)
- CMAKE_VERSION (standard 3.26.3)
- GMX_VERSION (standard 2023.1)

Set them when building with:

```bash
$ docker build --build-arg ENVIRONMENT_MODULES_VERSION="5.2.0" .
```
