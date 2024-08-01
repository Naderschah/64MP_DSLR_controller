FROM amd64/julia:latest
WORKDIR /FinalDataProcessing
# Install packages
RUN julia -e 'using Pkg; Pkg.add.(["DSP","PProf","JLD2","Images","StatsBase", "IterTools", "FilePathsBase"])'
# Install ps command
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*
CMD '/bin/bash'


# Needs to be run as:
# docker container rm julia_app; docker run -it -v /home/felix/OldUbuntu/home/felix/PiCamera_DSLR_like_controller:/FinalDataProcessing -v /run/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/:/Images -v /home/felix/RapidStorage2:/SaveSpot --name julia_app my_julia_image
# With build
# docker build -t my_julia_image .
