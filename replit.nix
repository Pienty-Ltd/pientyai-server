{pkgs}: {
  deps = [
    pkgs.zlib
    pkgs.xcodebuild
    pkgs.jq
    pkgs.glibcLocales
    pkgs.pkg-config
    pkgs.libffi
    pkgs.cacert
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
