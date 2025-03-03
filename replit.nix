{pkgs}: {
  deps = [
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.procps
    pkgs.lsof
    pkgs.libxcrypt
    pkgs.postgresql
    pkgs.openssl
  ];
}
