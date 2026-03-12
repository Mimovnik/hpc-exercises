{
  description = "OpenMPI development shell template";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem = {pkgs, ...}: {
        devShells.default = pkgs.mkShellNoCC {
          buildInputs = with pkgs; [
            gcc
            mpi
            mpi.dev
          ];

          shellHook = ''
            echo "Welcome to OpenMPI devshell!"

            # Generate clangd configuration so it knows where mpi.h lives.
            cat > .clangd <<EOF
            CompileFlags:
              Add:
                - "-I${pkgs.mpi.dev}/include"
            EOF
          '';
        };
      };
    };
}
