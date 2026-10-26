# docker build -f docker/sail.Dockerfile --target runtime -t sglang:sail .
# docker build -f docker/sail.Dockerfile --target framework -t sglang:sail-dev .
#
# SAIL (T-HEAD PPU) image. Ported from docker/ppu.Dockerfile on release/v0.5.21
# and adapted to the v0.5.22 layout:
#   - the platform pyproject is python/pyproject_other.toml (extras all_sail /
#     srt_sail), there is no dedicated pyproject_sail.toml copy;
#   - the AOT kernel is built from python/sglang/kernels/aot/setup_sail.py,
#     which targets the PPU SDK nvcc and registers the SM80 op surface;
#   - Rust extensions moved to ../rust and are filtered by the platform
#     pyproject's [tool.sglang] rust-extensions, so only "multimodal" is built;
#   - `kernels lock/download` (kernels-community cubins) is skipped: those are
#     CUDA builds, SAIL uses the PPU wheels from the flytiger index instead.
#
# Dropped w.r.t. docker/Dockerfile (CUDA) because they are NVIDIA-only or
# replaced by PPU builds: GDRCopy, HPC-Ops, MSCCL++, Mooncake, nixl, ai-dynamo,
# cosmos-guardrail, flashinfer-cubin/jit-cache, sgl-deep-ep, sgl-deep-gemm,
# NCCL/NVSHMEM.
#
# The base image ships the SAIL torch build in its own interpreter, so packages
# are installed into that interpreter (PIP_BREAK_SYSTEM_PACKAGES=1) instead of
# the /opt/sglang venv used by docker/Dockerfile: a fresh venv would hide SAIL
# torch and the resolver would pull a CUDA torch from PyPI. Every source build
# therefore uses --no-build-isolation.
#
# Precondition: the SAIL port is in the tree (python/pyproject_other.toml sail
# extras, python/sglang/kernels/aot/setup_sail.py and
# csrc/common_extension_sail.cc, python/sglang/srt/platforms/sail.py). On a
# pristine release branch the build stops at the stage that consumes them.
#
# Global ARGs: redeclare `ARG NAME` (without a value) inside a stage to use them.
ARG SAIL_BASE_IMAGE=pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.13.0-ubuntu24.04-cuda13.0-py312-20261001
# Platform extra from python/pyproject_other.toml. `srt_sail` is the SAIL
# runtime set; use `all_sail` (or `srt_sail,ray,tracing`) for the full image.
ARG BUILD_TYPE=srt_sail
ARG BRANCH_TYPE=local
ARG SGL_VERSION
ARG SGLANG_REPO=https://github.com/flytiger-eco/sglang-for-sail.git
ARG PIP_DEFAULT_INDEX
ARG SAIL_PIP_INDEX=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
ARG UBUNTU_MIRROR
ARG MAX_JOBS=16

# Must match the version in python/sglang/kernels/aot (pyproject*.toml) and the
# sglang-kernel requirements of the platform extra.
ARG SGL_KERNEL_VERSION=0.4.9

FROM ${SAIL_BASE_IMAGE} AS base

ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR

ARG PPU_PACKAGES="\
    acext==1.1.0+v0.2.0.ppu2.2.0 \
    cpm-kernels==1.0.11+v0.1.0.ppu2.2.0 \
    cuda-bindings==13.0.0+v0.1.0.ppu2.2.0 \
    cuda-python==13.0.0+v0.1.0.ppu2.2.0 \
    deep_ep==2.2.0+v0.1.0.ppu2.2.0 \
    deep_gemm==1.1.0+v0.1.0.ppu2.2.0 \
    fast-hadamard-transform==1.0.4.post1+v0.1.0.ppu2.2.0 \
    flash-attn==2.8.2+v0.2.0.ppu2.2.0 \
    flash_attn_3==2.8.2+v0.2.0.ppu2.2.0 \
    flash_mla==2.0.0+v0.2.0.ppu2.2.0 \
    flashinfer_python==0.6.17+v0.1.0.ppu2.2.0 \
    mooncake==0.3.12.post1+v0.1.0.ppu2.2.0 \
    nvidia-dali-cuda130==1.53.0+v0.1.0.ppu2.2.0 \
    pla==1.0.0+v0.1.0.ppu2.2.0 \
    tilelang==0.1.11+v0.1.0.ppu2.2.0 \
    torch_memory_saver==0.0.9+v0.1.0.ppu2.2.0 \
    torchao==0.17.0+v0.1.0.ppu2.2.0 \
    torchcodec==0.15.0+v0.1.0.ppu2.2.0 \
    transformer-engine==2.5+v0.1.0.ppu2.2.0 \
    triton==3.7.0+v0.1.0.ppu2.2.0"

ARG PPU_TORCH_PACKAGES="torchaudio==2.11.0+v0.1.0.ppu2.2.0"

# The base image ships torch in its own interpreter, so packages are installed
# into that interpreter instead of a separate venv (a venv would hide torch).
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

USER root

# Replace Ubuntu sources if specified (24.04 uses deb822 ubuntu.sources)
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
        [ -f "$f" ] && sed -i -e "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" \
                              -e "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" "$f"; \
    done; \
fi

# Install system dependencies (organized by category for better caching)
RUN --mount=type=cache,target=/var/cache/apt,id=sail-base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    # Core system utilities
    ca-certificates \
    netcat-openbsd \
    unzip \
    curl \
    wget \
    git \
    git-lfs \
    lsof \
    locales \
    # Build essentials (needed to compile the AOT kernel and sglang Rust extensions)
    build-essential \
    cmake \
    pkg-config \
    patchelf \
    ccache \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev \
    # NUMA
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM / opencv runtime
    ffmpeg \
    libgl1 \
    libglib2.0-0t64 \
    # InfiniBand/RDMA (DeepEP inter-node)
    libibverbs1 \
    libibverbs-dev \
    librdmacm1 \
    libibumad3 \
    ibverbs-providers \
    rdma-core \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Replace pip global index if specified
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    python3 -m pip config set global.index-url ${PIP_DEFAULT_INDEX}; \
fi

# Build backends for the SAIL kernel and sglang itself. All of them are built
# with --no-build-isolation so they compile against the SAIL torch already in
# this image (an isolated build env would pull the CUDA torch pinned by the
# in-tree pyproject from PyPI instead).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip uv \
    && python3 -m pip install \
        "setuptools>=75" \
        "setuptools-rust>=1.11" \
        "setuptools-scm>=8" \
        wheel \
        packaging \
        ninja \
        pybind11 \
        cython \
        scikit-build-core \
    && python3 -c "import torch; print('base torch:', torch.__version__)"

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
########## PARALLEL BUILDER STAGES ####################
########################################################
#
# These stages run IN PARALLEL via BuildKit:
#
#   base
#     |
#     +-- sail_kernels_builder (build sglang-kernel into wheels)
#     +-- gateway_builder      (independent, only needs gateway source)
#     |
#     v
#   sail_deps -> sglang_install -> framework / runtime
#

########################################################
# PARALLEL STAGE 1: SAIL Kernels Builder (starts from base)
########################################################
FROM base AS sail_kernels_builder

ARG SAIL_PIP_INDEX
ARG MAX_JOBS
ARG PPU_PACKAGES
ARG PPU_TORCH_PACKAGES
ARG SGL_KERNEL_VERSION

WORKDIR /build

# Build one package per pip call so a failure points at the offending sdist.
# No pip cache here: some shim sdists download a prebuilt wheel and os.rename()
# it into pip's output dir, which fails with EXDEV when that dir is on a cache
# mount. --no-cache-dir keeps the output dir on the same filesystem.
RUN mkdir -p /wheels \
    && for spec in ${PPU_PACKAGES} ${PPU_TORCH_PACKAGES}; do \
        echo "==> Building ${spec}" \
        && MAX_JOBS=${MAX_JOBS} python3 -m pip wheel \
            --no-cache-dir \
            --no-deps \
            --no-build-isolation \
            --index-url ${SAIL_PIP_INDEX} \
            --wheel-dir /wheels \
            "${spec}" \
        || exit 1; \
    done \
    && ls -l /wheels

# Prefer the published PPU kernel. If the requested version has not been
# published yet, build it with setup_sail.py (nvcc from the PPU SDK, SM80 op
# surface). The wheel is versioned by the in-tree manifest, so verify the pin
# before spending the build time.
# setup_sail.py reaches into ../jit/include, so keep the aot/jit layout and
# expect the source build to clone actlize/flashinfer/triton at build time.
COPY python/sglang/kernels/aot /build/sglang/kernels/aot
COPY python/sglang/kernels/jit/include /build/sglang/kernels/jit/include
RUN if MAX_JOBS=${MAX_JOBS} python3 -m pip wheel \
        --no-cache-dir \
        --no-deps \
        --no-build-isolation \
        --index-url ${SAIL_PIP_INDEX} \
        --wheel-dir /wheels \
        "sglang-kernel==${SGL_KERNEL_VERSION}"; then \
        echo "Using published sglang-kernel ${SGL_KERNEL_VERSION}"; \
    else \
        echo "flytiger has no sglang-kernel ${SGL_KERNEL_VERSION}; building from AOT sources" \
        && cd /build/sglang/kernels/aot \
        && { grep -h '^version = ' pyproject_sail.toml pyproject.toml 2>/dev/null \
                | grep -qx "version = \"${SGL_KERNEL_VERSION}\"" \
            || { echo "kernels/aot is not sglang-kernel ${SGL_KERNEL_VERSION}"; exit 1; }; } \
        && MAX_JOBS=${MAX_JOBS} python3 setup_sail.py bdist_wheel --dist-dir /wheels; \
    fi \
    && rm -rf /build/sglang \
    && ls -l /wheels

########################################################
# PARALLEL STAGE 2: Gateway Builder (starts from base)
########################################################
# Builds sgl-model-gateway in isolation so Python-only changes
# don't trigger a full Rust recompilation.
FROM base AS gateway_builder

WORKDIR /build

# Copy ONLY the gateway source (not the full repo)
COPY sgl-model-gateway /build/sgl-model-gateway

# Install Rust, build gateway binary and Python bindings, then clean up Rust toolchain
RUN --mount=type=cache,target=/root/.cache/pip \
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && python3 -m pip install maturin \
    && cd /build/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out /build/gateway_wheels \
    && cd /build/sgl-model-gateway \
    && cargo build --release --bin sgl-model-gateway --features vendored-openssl \
    && cp target/release/sgl-model-gateway /build/sgl-model-gateway-bin \
    && rm -rf /root/.cargo /root/.rustup /build/sgl-model-gateway/target /build/sgl-model-gateway/bindings/python/target

########################################################
# STAGE 3: SAIL Deps (needs sail_kernels_builder)
########################################################
FROM base AS sail_deps

ARG BUILD_TYPE
ARG SAIL_PIP_INDEX
ARG PPU_PACKAGES
ARG PPU_TORCH_PACKAGES

WORKDIR /sgl-workspace

# Install the locally built SAIL kernel and the router from the builder stages.
# --no-deps keeps pip from resolving requirements against PyPI here; they are
# resolved together with the sglang dependencies below.
COPY --from=sail_kernels_builder /wheels /tmp/wheels/sail
COPY --from=gateway_builder /build/gateway_wheels /tmp/wheels/gateway
RUN python3 -m pip install --no-deps /tmp/wheels/sail/*.whl /tmp/wheels/gateway/*.whl \
    && rm -rf /tmp/wheels

# Pin torch from the base image and the SAIL builds so the dependency solve
# below can never replace them with CUDA wheels from PyPI. The base image may
# expose its torch as `torch` or as the SAIL distribution (torch-for-sail);
# both spellings are pinned when present.
RUN python3 -m pip list --format=freeze \
        | grep -iE '^(torch|torchvision|torchaudio|triton|torch[-_]for[-_]sail|triton[-_]for[-_]sail|sglang-router|sglang[-_]kernel)==' > /sgl-workspace/sail_constraints.txt \
    && for spec in ${PPU_PACKAGES} ${PPU_TORCH_PACKAGES}; do echo "${spec}"; done \
        >> /sgl-workspace/sail_constraints.txt \
    && cat /sgl-workspace/sail_constraints.txt

# Install sglang dependencies from a stub project built from the platform
# pyproject. Split from the editable install so Python source changes don't
# invalidate the dep-install layer. No setup.py is copied, so no Rust extension
# is built here. Generate constraints.txt to prevent reinstalling these deps in
# later stages.
# uv is used instead of pip: pip's backtracking on this dependency set falls
# back to building published shim sdists (which fail outside the builder),
# whereas uv resolves it directly. --only-binary on the pinned packages makes
# any such fallback fail loudly instead.
COPY python/pyproject_other.toml /tmp/sglang_deps/python/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    cd /tmp/sglang_deps/python \
    && mkdir -p sglang \
    && touch sglang/__init__.py \
    && touch README.md \
    && touch LICENSE \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install --system --break-system-packages \
        $(grep '==' /sgl-workspace/sail_constraints.txt | cut -d= -f1 | sed 's/^/--only-binary /') \
        -c /sgl-workspace/sail_constraints.txt \
        --index-strategy unsafe-best-match \
        --extra-index-url ${SAIL_PIP_INDEX} \
        ".[${BUILD_TYPE}]" \
    && python3 -m pip uninstall -y sglang \
    && cd /sgl-workspace \
    && rm -rf /tmp/sglang_deps \
    && python3 -m pip list --format=freeze | grep -v "^sglang==" > /sgl-workspace/constraints.txt

########################################################
# STAGE 4: SGLang Install (needs sail_deps)
########################################################

# Copy local source if building from local
FROM scratch AS local_src
COPY . /src

FROM sail_deps AS sglang_install

ARG BRANCH_TYPE
ARG BUILD_TYPE
ARG SGL_VERSION
ARG SGLANG_REPO

WORKDIR /sgl-workspace

COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /sgl-workspace/sglang; \
    elif [ -z "$SGL_VERSION" ]; then \
        git clone --depth=1 ${SGLANG_REPO} /sgl-workspace/sglang; \
    else \
        git clone --depth=1 --branch v${SGL_VERSION} ${SGLANG_REPO} /sgl-workspace/sglang; \
    fi \
    && rm -rf /tmp/local_src

# Editable install (fast - dependencies already installed via constraints).
# The platform pyproject replaces python/pyproject.toml, which also selects the
# Rust extensions to build (../rust, multimodal only) and keeps the
# torch==2.14.1 pin of the in-tree pyproject out of the build environment. The
# toolchain and cargo target dir are removed in the same layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cargo/registry \
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && cd /sgl-workspace/sglang \
    && cp python/pyproject_other.toml python/pyproject.toml \
    && CARGO_TARGET_DIR=/tmp/cargo-target python3 -m pip install \
        --no-deps --no-build-isolation -e "python[${BUILD_TYPE}]" \
    && rm -rf /tmp/cargo-target /root/.rustup \
    && find /root/.cargo -mindepth 1 -maxdepth 1 ! -name registry -exec rm -rf {} + \
    && mkdir -p /root/.cache/huggingface /root/.cache/sglang \
    && python3 -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"

# TODO(sail): kernels-community/sgl-flash-attn3 cubins (`kernels lock/download`)
# are CUDA builds; SAIL uses flash_attn_3 from the flytiger index instead.

ENV SGLANG_RUST_BUILD_MODE=never

########################################################
########## Framework Image (dev) ######################
########################################################
FROM sglang_install AS framework

WORKDIR /sgl-workspace

# Install dev apt packages
RUN --mount=type=cache,target=/var/cache/apt,id=sail-framework-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    zsh \
    tree \
    silversearcher-ag \
    less \
    infiniband-diags \
    perftest \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Rust toolchain for rebuilding sglang Rust extensions during development.
# Requires >= 1.85 (edition 2024).
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

# Install essential Python packages (use constraints to prevent conflicts)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -c /sgl-workspace/constraints.txt \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    py-spy \
    pandas \
    matplotlib \
    tabulate

# These configs are optional; users can override them by mounting their own files
COPY docker/configs/opt/.vimrc /root/.vimrc
COPY docker/configs/opt/.tmux.conf /root/.tmux.conf

# Install the pre-built gateway binary; its Python wheel is installed in sail_deps.
COPY --from=gateway_builder /build/sgl-model-gateway-bin /usr/local/bin/sgl-model-gateway

WORKDIR /sgl-workspace/sglang

CMD ["/bin/bash"]

########################################################
########## Runtime Image ##############################
########################################################
#
# PURPOSE: Production runtime environment. The PPU base image already carries
# the SDK toolchain needed for JIT, so runtime simply reuses sglang_install
# without the dev tools added in framework.
FROM sglang_install AS runtime

# Install the pre-built gateway binary; its Python wheel is installed in sail_deps.
COPY --from=gateway_builder /build/sgl-model-gateway-bin /usr/local/bin/sgl-model-gateway

WORKDIR /sgl-workspace/sglang

# Keep build provenance at the end so metadata changes do not invalidate build layers.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang:sail
ENV SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang:sail}
LABEL org.opencontainers.image.source="https://github.com/flytiger-eco/sglang-for-sail" \
      org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}" \
      org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      org.opencontainers.image.url="${SGLANG_BUILD_URL}" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.build.url="${SGLANG_BUILD_URL}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

# Default command
CMD ["/bin/bash"]
