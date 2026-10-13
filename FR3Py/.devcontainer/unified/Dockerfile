ARG SHARPA_IMAGE=sharpadev/sharpawavesdk:5.0.2.7f0e
FROM ${SHARPA_IMAGE}

# Opt-in build-time download of the Sharpa Wave SDK Debian package. Leave
# INSTALL_SHARPA_SDK at 0 to use the SDK bundled in the base image.
ARG INSTALL_SHARPA_SDK=0
ARG SHARPA_SDK_URL=https://github.com/sharpa-robotics/sharpa-wave-sdk/releases/download/v5.0.10/sharpa-wave-sdk-cuda_5.0.10_amd64.deb
ARG SHARPA_SDK_SHA256=2e2cacd5f7e136f6d5513e828f63e540a293210d843e6103bbe14ff967fa5e68

ENV DEBIAN_FRONTEND=noninteractive
# The base image is Ubuntu 24.04, whose system interpreter is marked as
# externally managed (PEP 668). This container has a single python, so let pip
# install into it both at build time and from the devcontainer postCreate step.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        libpoco-dev \
        libeigen3-dev \
        libyaml-cpp-dev \
        libglvnd-dev \
        libgl1-mesa-dev \
        libglib2.0-dev \
        python3-dev \
        python3-tk \
        libevdev-dev \
    && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    if [ "${INSTALL_SHARPA_SDK}" != "1" ]; then \
        echo "INSTALL_SHARPA_SDK=${INSTALL_SHARPA_SDK}, skipping the private OpenCV 4.5 runtime libs."; \
    else \
        mkdir -p /tmp/ocv45/extract /opt/opencv45/lib; \
        curl -fL \
            -o /tmp/ocv45/libopencv-core4.5d_4.5.4+dfsg-9ubuntu4_amd64.deb \
            http://archive.ubuntu.com/ubuntu/pool/universe/o/opencv/libopencv-core4.5d_4.5.4+dfsg-9ubuntu4_amd64.deb; \
        curl -fL \
            -o /tmp/ocv45/libopencv-imgproc4.5d_4.5.4+dfsg-9ubuntu4_amd64.deb \
            http://archive.ubuntu.com/ubuntu/pool/universe/o/opencv/libopencv-imgproc4.5d_4.5.4+dfsg-9ubuntu4_amd64.deb; \
        curl -fL \
            -o /tmp/ocv45/libtbb2_2020.3-1ubuntu3_amd64.deb \
            http://archive.ubuntu.com/ubuntu/pool/universe/t/tbb/libtbb2_2020.3-1ubuntu3_amd64.deb; \
        dpkg-deb -x /tmp/ocv45/libopencv-core4.5d_4.5.4+dfsg-9ubuntu4_amd64.deb /tmp/ocv45/extract; \
        dpkg-deb -x /tmp/ocv45/libopencv-imgproc4.5d_4.5.4+dfsg-9ubuntu4_amd64.deb /tmp/ocv45/extract; \
        dpkg-deb -x /tmp/ocv45/libtbb2_2020.3-1ubuntu3_amd64.deb /tmp/ocv45/extract; \
        cp -a /tmp/ocv45/extract/usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5* \
              /tmp/ocv45/extract/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5* \
              /tmp/ocv45/extract/usr/lib/x86_64-linux-gnu/libtbb.so.2* \
              /opt/opencv45/lib/; \
        rm -rf /tmp/ocv45; \
    fi

RUN set -eu; \
    if [ "${INSTALL_SHARPA_SDK}" != "1" ]; then \
        echo "INSTALL_SHARPA_SDK=${INSTALL_SHARPA_SDK}, using the SDK bundled in the base image."; \
    else \
        echo "Downloading the Sharpa Wave SDK from ${SHARPA_SDK_URL}"; \
        curl -fL "${SHARPA_SDK_URL}" -o /tmp/sharpa-sdk.deb; \
        echo "${SHARPA_SDK_SHA256}  /tmp/sharpa-sdk.deb" | sha256sum -c -; \
        apt-get update && apt-get install -y /tmp/sharpa-sdk.deb; \
        rm -f /tmp/sharpa-sdk.deb; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# The SDK installed above ships its own libsharpa-wave-sdk.so; it must win over
# the older copy in the base image, otherwise importing the python binding fails
# with an undefined symbol. The private OpenCV 4.5 prefix supplies runtime libs
# the SDK links against without touching the system OpenCV.
ENV LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:/opt/opencv45/lib:/usr/local/lib:${LD_LIBRARY_PATH}

WORKDIR /opt
RUN git clone https://github.com/lcm-proj/lcm.git \
    && cd lcm && mkdir build && cd build \
    && PYTHON_BIN="$(python3 -c 'import sys; print(sys.executable)')" \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLCM_ENABLE_PYTHON=ON \
        -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
        -DPython_EXECUTABLE="${PYTHON_BIN}" \
        -DPython3_EXECUTABLE="${PYTHON_BIN}" \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig

WORKDIR /opt
RUN git clone --recursive https://github.com/frankaemika/libfranka --branch 0.13.3 \
    && cd libfranka && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. \
    && cmake --build . -j"$(nproc)" \
    && cpack -G DEB \
    && dpkg -i libfranka*.deb \
    && ldconfig

RUN python3 -m pip install --no-cache-dir \
        numpy \
        pyzmq \
        pyyaml \
        pin \
        proxsuite \
        scipy \
        ipykernel \
        matplotlib \
        opencv-python \
        mujoco

# Make the `sharpa` package shipped with the SDK image importable from any
# interpreter by pointing a .pth file at the SDK's python directory.
RUN set -eu; \
    SDK_PY=""; \
    for candidate in /opt/sharpa-wave-sdk/python /root/sharpa-wave-sdk/python; do \
        if [ -d "$candidate" ]; then SDK_PY="$candidate"; break; fi; \
    done; \
    if [ -z "$SDK_PY" ]; then \
        echo "ERROR: the Sharpa Wave SDK python directory was not found." >&2; \
        echo "       Looked in /opt/sharpa-wave-sdk/python and /root/sharpa-wave-sdk/python." >&2; \
        echo "       Set SHARPA_IMAGE to an image that ships the SDK (e.g. sharpadev/sharpawavesdk:5.0.2.7f0e)," >&2; \
        echo "       or build with INSTALL_SHARPA_SDK=1 to install the SDK package." >&2; \
        exit 1; \
    fi; \
    SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    mkdir -p "$SITE_PACKAGES"; \
    echo "$SDK_PY" > "$SITE_PACKAGES/sharpa-wave-sdk.pth"; \
    echo "Sharpa Wave SDK python path ${SDK_PY} registered in ${SITE_PACKAGES}"

RUN set -eu; \
    python3 -c "import sharpa; print(sharpa.__file__)"; \
    if [ "${INSTALL_SHARPA_SDK}" = "1" ]; then \
        python3 -c "import sharpa, sys; f = sharpa.__file__; sys.exit(0 if f.startswith('/opt/sharpa-wave-sdk/') else 'expected the installed SDK under /opt/sharpa-wave-sdk/, got %s' % f)"; \
        dpkg-query -W -f='${Package} ${Version}\n' | grep -i sharpa; \
        EXT_SO=/opt/sharpa-wave-sdk/python/sharpa/sharpa.cpython-312-x86_64-linux-gnu.so; \
        ldd "$EXT_SO" | grep -i sharpa; \
        RESOLVED="$(ldd "$EXT_SO" | awk '$1 ~ /^libsharpa-wave-sdk\.so/ && $3 ~ /^\// {print $3; exit}')"; \
        case "${RESOLVED:-}" in \
            /opt/sharpa-wave-sdk/lib/*) \
                echo "libsharpa-wave-sdk.so resolves to ${RESOLVED}";; \
            *) \
                echo "ERROR: libsharpa-wave-sdk.so resolved to '${RESOLVED:-<unresolved>}'," >&2; \
                echo "       expected a path under /opt/sharpa-wave-sdk/lib." >&2; \
                exit 1;; \
        esac; \
    fi

# Interactive shells (e.g. devcontainer terminals) source /root/.bashrc, which
# the base image uses to set LD_LIBRARY_PATH; append after it so the installed
# SDK and private OpenCV 4.5 libs under /opt take precedence there too.
RUN set -eu; \
    if [ "${INSTALL_SHARPA_SDK}" = "1" ]; then \
        printf '\nexport LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:/opt/opencv45/lib:${LD_LIBRARY_PATH}\n' >> /root/.bashrc; \
    fi

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

WORKDIR /home/FR3Py
