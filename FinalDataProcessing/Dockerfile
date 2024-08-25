FROM amd64/julia:latest
WORKDIR /FinalDataProcessing
# Install packages
ENV JULIA_NUM_THREADS=8
RUN julia -e 'using Pkg; Pkg.add.(["DSP","PProf","JLD2","Images","StatsBase", "IterTools", "FilePathsBase", "HDF5","FixedPointNumbers", "ProgressMeter","PrettyTables","AbstractFFTs", "ImageTransformations","Rotations","CoordinateTransformations","ImageSegmentation","ImageFiltering","ImageQualityIndexes","Plots", "IJulia"])'
# Install ps command
RUN apt-get update && apt-get install -y procps
# So that IJulia Works 
RUN apt-get install -y python3-full python3-pip
# Get Jupyter (for IJulia), without making a seperate virtual environemtn
RUN pip install --break-system-packages jupyterlab notebook
# TODO: Find the alias that worked without
ENV jupyter-notebook="jupyter-notebook --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''"
CMD '/bin/bash'


# Needs to be run as: (-it interactive -p for notebook port -v all the directory bind mounts )
# docker container rm julia_app; docker run -it -p 8888:8888 -v /home/felix/Microscope/:/FinalDataProcessing -v /MicroscopeNetStorage/Images/:/Images -v /home/felix/RapidStorage2:/SaveSpot --name julia_app my_julia_image
# With build
# docker build -t my_julia_image .
