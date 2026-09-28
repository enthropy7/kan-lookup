{
  description = "ESP32 benchmark firmware (ESP-IDF 5.5.2)";
  inputs.esp-dev.url = "github:mirrexagon/nixpkgs-esp-dev/5287d6e1ca9e15ebd5113c41b9590c468e1e001b";
  # nix develop takes its bash from the flake's nixpkgs input; without one it looks nixpkgs up in the online
  # registry and, offline, falls back to the system bash (3.2 on macOS, too old for the shell script)
  inputs.nixpkgs.follows = "esp-dev/nixpkgs";
  outputs = { self, esp-dev, nixpkgs }: {
    devShells = builtins.listToAttrs (map (system: {
      name = system;
      value.default = esp-dev.devShells.${system}.esp32-idf;
    }) [ "x86_64-linux" "aarch64-darwin" ]);
  };
}
