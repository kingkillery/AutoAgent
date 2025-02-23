FROM tjbtech1/gaia-bookworm:v2

# Update package lists and install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Keep the rest of the original image configuration 