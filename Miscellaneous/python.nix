let
   pkgs = import <nixpkgs> {};
in pkgs.mkShell {
   packages = [
      (pkgs.python3.withPackages (python-pkgs: [
         python-pkgs.numpy
         python-pkgs.scipy
	      python-pkgs.numba
	      python-pkgs.matplotlib
	      python-pkgs.requests
	      python-pkgs.pip
	      python-pkgs.ipython
	      python-pkgs.jupyter
	      python-pkgs.h5py
	      python-pkgs.opencv4
	      python-pkgs.h5py
	      python-pkgs.pytest
	      python-pkgs.xarray
	      python-pkgs.colour
	      python-pkgs.beautifulsoup4
	      python-pkgs.scikit-learn
	      python-pkgs.scikit-image
	      python-pkgs.numba
	      python-pkgs.plotly
	      (python-pkgs.buildPythonPackage {
                pname = "mpl-interactions";
           	version = "0.24.1";
           	src = pkgs.fetchurl {
			url = "https://files.pythonhosted.org/packages/52/20/42ed756ee8d338e9cd8035f3648cd87494cb1094261f67edc725b4ab79d5/mpl_interactions-0.24.1.tar.gz";
             		sha256 = "0grv6vbfw8z1smwg273mgr87zqczznjd63j6r4drb680ns940yx7"; 
           		};
           	propagatedBuildInputs = [ python-pkgs.matplotlib python-pkgs.numpy python-pkgs.ipywidgets python-pkgs.xarray python-pkgs.pytest ];
         	})
        ]))
	pkgs.hdf5
   ];
   shellHook = ''
     alias jupyter-notebook="jupyter notebook --NotebookApp.token=\"\" --no-browser --NotebookApp.disable_check_xsrf=True"
   '';
}
