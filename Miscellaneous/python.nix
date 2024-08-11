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
        ]))
	pkgs.hdf5
   ];
}
