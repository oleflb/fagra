{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      cargoToml = fromTOML (builtins.readFile ./Cargo.toml);
      crate = cargoToml.package;

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems =
        function:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          function {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      packages = forAllSystems (
        { pkgs }:
        {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = crate.name;
            inherit (crate) version;

            src = pkgs.lib.cleanSource ./.;
            cargoLock.lockFile = ./Cargo.lock;

            strictDeps = true;

            meta =
              {
                platforms = pkgs.lib.platforms.unix;
              }
              // pkgs.lib.optionalAttrs (crate ? description) {
                inherit (crate) description;
              }
              // pkgs.lib.optionalAttrs (crate ? homepage) {
                inherit (crate) homepage;
              };
          };
        }
      );

      checks = forAllSystems (
        { pkgs }:
        {
          package = self.packages.${pkgs.system}.default;
        }
      );

      devShells = forAllSystems (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              clippy
              rust-analyzer
              rustc
              rustfmt
            ];

            RUST_BACKTRACE = "1";
          };
        }
      );

      formatter = forAllSystems ({ pkgs }: pkgs.nixfmt-rfc-style);
    };
}
