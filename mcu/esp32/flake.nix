{
  description = "ESP32 benchmark firmware (ESP-IDF 5.5.2)";
  inputs.esp-dev.url = "github:mirrexagon/nixpkgs-esp-dev/5287d6e1ca9e15ebd5113c41b9590c468e1e001b";
  outputs = { self, esp-dev }: {
    devShells = builtins.listToAttrs (map (system: {
      name = system;
      value.default = esp-dev.devShells.${system}.esp32-idf;
    }) [ "x86_64-linux" "aarch64-darwin" ]);
  };
}
