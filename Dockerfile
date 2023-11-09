FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-08-16

ARG REPO_BRANCH="main"

WORKDIR /repo

# Copy pyproject.toml first so that we done need to reinstall in case anoter file
# is changing ater rebuiding docker image
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/finsberg/stress-fem-vs-simple.git
RUN cd stress-fem-vs-simple && python3 -m pip install pip --upgrade && python3 -m pip install --no-cache-dir -r requirements.txt && rm -rf /tmp
