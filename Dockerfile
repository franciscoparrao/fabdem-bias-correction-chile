FROM mambaorg/micromamba:1.5-jammy

LABEL maintainer="francisco.parra.o@usach.cl"
LABEL description="FABDEM bias correction pipeline (Parra Ortiz, ISPRS J P&RS submission)"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# SurtGis (Rust geospatial library) — required for terrain + hydrology stages
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && /root/.cargo/bin/cargo install --git https://github.com/franciscoparrao/surtgis --bin surtgis
ENV PATH="/root/.cargo/bin:${PATH}"

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /home/$MAMBA_USER/work
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

ENV PROJ_NETWORK=ON

CMD ["bash"]
