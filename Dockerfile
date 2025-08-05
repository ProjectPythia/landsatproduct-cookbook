FROM quay.io/cryointhecloud/cryo-hub-image:25e9cee21f58
COPY environment.yml /tmp/environment.yml
RUN mamba env update --prefix ${CONDA_DIR} --file /tmp/environment.yml
RUN ls