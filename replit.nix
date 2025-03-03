{pkgs}: {
  deps = [
    pkgs.procps
    pkgs.lsof
    pkgs.libxcrypt
    pkgs.postgresql
    pkgs.openssl
  ];
}
