{
  description = "KAN tables for firmware";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = (pkgs.python312.override {
        packageOverrides = pyself: pysuper: {
          # CPU build of the PyTorch release the results were produced with
          torch = pyself.buildPythonPackage {
            pname = "torch"; version = "2.13.0"; format = "wheel";
            src = pkgs.fetchurl {
              url = "https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl";
              sha256 = "4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2";
            };
            nativeBuildInputs = [ pkgs.autoPatchelfHook ];
            buildInputs = [ pkgs.stdenv.cc.cc.lib ];
            propagatedBuildInputs = with pyself; [ filelock typing-extensions sympy networkx jinja2 fsspec setuptools numpy ];
            doCheck = false;
          };
          # their test suites fail in the build sandbox
          narwhals = pysuper.narwhals.overridePythonAttrs (_: { doCheck = false; });
          pvlib = pysuper.pvlib.overridePythonAttrs (_: { doCheck = false; });
          commonroad-vehicle-models = pyself.buildPythonPackage {
            pname = "commonroad-vehicle-models"; version = "3.0.2"; format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/c3/05/0bcea75e39bcb98192bfe908d2e53d636f87d0d9a365dcd95ffe5c1d9373/commonroad_vehicle_models-3.0.2-py3-none-any.whl";
              sha256 = "c8a675ddfdbba0c14521a4479cd820cfb42d03eb549723b00bd9384a434ddd6e";
            };
            propagatedBuildInputs = [ pyself.omegaconf pyself.numpy ];
            doCheck = false;
          };
          pymsis = pyself.buildPythonPackage {
            pname = "pymsis"; version = "0.13.0"; format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/0f/37/d42cd57f947dcb3c7bbe7b0a2148b31ed37d832f000dd5b6a81ef34231f8/pymsis-0.13.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
              sha256 = "f58eb21787ef9e9585d1987c5f494d4e9f83a96e2daa0b47f705d76628135925";
            };
            nativeBuildInputs = [ pkgs.autoPatchelfHook ];
            buildInputs = [ pkgs.gfortran.cc.lib pkgs.zlib ];
            propagatedBuildInputs = [ pyself.numpy ];
            doCheck = false;
          };
          coolprop = pyself.buildPythonPackage {
            pname = "coolprop"; version = "8.0.0"; format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/6e/c3/c2d3630212d31677ac17b917a12014fb0efdfd3e4494a62f38fc40c530e2/coolprop-8.0.0-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
              sha256 = "8ca1aefd1873b14f3e7e9122a08a32fe8c62ac03a1fc4496512a31a3bba8f611";
            };
            nativeBuildInputs = [ pkgs.autoPatchelfHook ];
            buildInputs = [ pkgs.stdenv.cc.cc.lib ];
            propagatedBuildInputs = [ pyself.numpy ];
            doCheck = false;
          };
          ppigrf = pyself.buildPythonPackage {
            pname = "ppigrf"; version = "2.1.0"; format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/c5/c8/7c561c46a39ac4a21c0cf2c104bf81f76ef2183fa69c0c2bb7c583d52f69/ppigrf-2.1.0-py3-none-any.whl";
              sha256 = "50ff8968380a90c50cc6085122ac4673b0890939e37efa375f24d5bee7b1e1d6";
            };
            propagatedBuildInputs = [ pyself.numpy pyself.pandas ];
            doCheck = false;
          };
          cantera = pyself.buildPythonPackage {
            pname = "cantera"; version = "3.2.0"; format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/2d/b1/5a89f0a6fa0bcab611564a02e86dea6b95292cab1da7370c35732eeeb596/cantera-3.2.0-cp312-cp312-manylinux_2_28_x86_64.whl";
              sha256 = "d7232fd69dda04b350b3d095dd5be234e4f627b8368421f8d1f976956feb3441";
            };
            nativeBuildInputs = [ pkgs.autoPatchelfHook ];
            buildInputs = [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
            propagatedBuildInputs = [ pyself.numpy pyself.ruamel-yaml pyself.typing-extensions ];
            doCheck = false;
          };
        };
      }).withPackages (ps: [ ps.torch ps.pytest ps.scikit-learn ps.matplotlib ps.openpyxl ps.xlrd ps.pvlib ps.commonroad-vehicle-models ps.pymsis ps.cantera ps.coolprop ps.ppigrf ]);

      uciZip = name: url: sha256: pkgs.runCommand name { nativeBuildInputs = [ pkgs.unzip ]; } ''
        mkdir $out && cd $out && unzip -q ${pkgs.fetchurl { inherit url sha256; }}
      '';
      ccpp = uciZip "uci-ccpp" "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
        "0ylcp4r69bvh3k3fbxyghs0wdcs6axz9l7f94614x960fx4jlyyc";
      airfoil = uciZip "uci-airfoil" "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"
        "0z70ga59x2hikv4w82ziz0nqkx0p851vj7ms90zpv0mdafx6fxsw";
      concrete = uciZip "uci-concrete" "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
        "19y9qa8wy1ssr693n6mp2lvk36jndd77galx8w3lxvlavqa5vn6s";
      energy = uciZip "uci-energy" "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
        "0lc3fckbdz17vp80qlkii95w371wqvc8zx8p0jqa8abrwbp43529";
      airQuality = uciZip "uci-air-quality" "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
        "0ala5fmhi37bix2qsvyzpldjx6q7lwy1k7cdljl8hliqzc9l19nl";
      # LG 18650HG2 cell tests, Kollmeyer et al., CC BY 4.0
      lgHg2 = pkgs.runCommand "lg-hg2-cp3473x7xv-v3" { nativeBuildInputs = [ pkgs.unzip ]; } ''
        mkdir $out && cd $out
        unzip -q ${pkgs.fetchurl {
          name = "lg-hg2-cp3473x7xv-v3.zip";
          url = "https://data.mendeley.com/public-api/zip/cp3473x7xv/download/3";
          sha256 = "0qlbzw5317v4xjlpd0gy56lz682qhvxy4fz4ca27n9lc4glsnpvj";
        }}
        mkdir prepared
        find . -name 'LGHG2@*.zip' -exec unzip -q {} -d prepared \;
      '';

      efficientKan = pkgs.fetchFromGitHub {
        owner = "Blealtan";
        repo = "efficient-kan";
        rev = "7b6ce1c87f18c8bc90c208f6b494042344216b11";
        hash = "sha256-DPvTnRbpN1agih6CLF+ckNCzGAJrkhVIHlttzsKjqM8=";
      };

      env = {
        KANTAB_CCPP = ccpp;
        KANTAB_AIRFOIL = airfoil;
        KANTAB_CONCRETE = concrete;
        KANTAB_ENERGY = energy;
        KANTAB_AIR_QUALITY = airQuality;
        KANTAB_LG_HG2 = lgHg2;
        KANTAB_EFFICIENT_KAN = "${efficientKan}/src";
        KANTAB_NIXPKGS_REV = nixpkgs.rev;
        PYTHONHASHSEED = "0";
        PYTHONDONTWRITEBYTECODE = "1";
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell ({
        # zig cross-compiles c/ for the boards
        packages = [ python pkgs.zig ];
      } // env);

      # every study's verdicts and the figures, recomputed from results/raw
      packages.${system}.default = pkgs.runCommand "kan-lookup-results" ({ nativeBuildInputs = [ python ]; } // env) ''
        cp -r ${self} src && chmod -R u+w src && cd src
        export HOME=$TMPDIR MPLCONFIGDIR=$TMPDIR
        python -m kantab.report --out $out
      '';

      # nix run: the verdicts in the terminal, the report in the browser
      apps.${system}.default = {
        type = "app";
        program = toString (pkgs.writeShellScript "kan-lookup-results" ''
          r=${self.packages.${system}.default}
          grep -E "^== |CONFIRMED|FALSIFIED|needs ≥" $r/verdicts.txt
          echo; echo "report: $r/index.html"
          ${pkgs.xdg-utils}/bin/xdg-open $r/index.html > /dev/null 2>&1 &
        '');
      };
    };
}
