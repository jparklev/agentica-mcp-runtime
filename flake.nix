{
  description = "Agentica - MCP tools as Python functions in a sandboxed REPL";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python313;

      # Common native libraries needed by Python packages with C extensions.
      # Without these on LD_LIBRARY_PATH, `import` of pip-installed packages
      # like cryptography, numpy, pandas, etc. will fail on NixOS.
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        openssl
        libffi
      ];

      pipWrapper = pkgs.writeShellScriptBin "pip" ''
        exec uv pip "$@"
      '';
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          python
          pkgs.uv
          pkgs.nodejs # provides `node` and `npx` for Node-based MCP servers
          pkgs.ty # LSP
          pkgs.ruff # Formatter
          pipWrapper  # `pip` that delegates to `uv pip` (works in subprocesses)
        ];

        env = {
          UV_PYTHON = "${python}/bin/python";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        };

        shellHook = ''
          unset PYTHONPATH
        '';
      };
    });
}
