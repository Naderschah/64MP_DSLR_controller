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
        ]))
   ];
}
