{
  description = "STM32F4 benchmark firmware";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/6774f7bc253789b113a4f39285dc0fa100abeacc";
  outputs = { self, nixpkgs }:
    let
      shell = system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          gh = owner: repo: rev: hash: pkgs.fetchFromGitHub { inherit owner repo rev hash; };
          cmsisNN = gh "ARM-software" "CMSIS-NN" "71cbe3d5a686c875c88a4d112633b13fae94b5a2" "sha256-Y6b6uMLoqKI9r1mF1C+TQcFF/y3xRRgWgqfV90c1JaM=";
          cmsisDSP = gh "ARM-software" "CMSIS-DSP" "83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b" "sha256-s96zRD3+cGRXJvBx8me34Rf27B1dkUe0lG8Sd9qOKqQ=";
          cmsisCore = gh "ARM-software" "CMSIS_6" "26206e47dcf0abfbdc64eb753a0b6334b24439f6" "sha256-rmJCTln4xXv61wtimPl1d+I/Fc2ozpM384XLCOW5wxg=";
          opencm3 = pkgs.stdenv.mkDerivation {
            name = "libopencm3-stm32f4";
            src = gh "libopencm3" "libopencm3" "2da12dc96e0b9e42a3332348dd9b02a0a17981f8" "sha256-qaFyxylwXPe0n+dOK5WL1vcFg1aDKHwxEWxUvJY5nKo=";
            nativeBuildInputs = [ pkgs.gcc-arm-embedded pkgs.python3 ];
            postPatch = "patchShebangs scripts";
            buildPhase = "make TARGETS=stm32/f4 -j$NIX_BUILD_CORES";
            installPhase = "mkdir -p $out && cp -r include lib $out/";
            dontFixup = true;
          };
        in pkgs.mkShell {
          packages = [ pkgs.gcc-arm-embedded pkgs.dfu-util pkgs.gnumake (pkgs.python312.withPackages (ps: [ ps.numpy ps.pyserial ps.pyusb ])) ];
          CMSIS_NN = cmsisNN;
          CMSIS_DSP = cmsisDSP;
          CMSIS_CORE = "${cmsisCore}/CMSIS/Core";
          OPENCM3 = opencm3;
        };
    in {
      devShells = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-darwin" ] (system: { default = shell system; });
    };
}
