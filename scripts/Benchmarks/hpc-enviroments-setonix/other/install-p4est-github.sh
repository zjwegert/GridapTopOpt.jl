# This script is specifically to build P4est from Github sources.
# This is necessary when installing P4est 2.2 with newer versions of Intel compilers.
#
# Steps:
#   - The key is that we require a newer version of autotools, which is unfortunately
#     not available in Gadi.
#     Install instructions [here](https://docs.open-mpi.org/en/v5.0.x/developers/gnu-autotools.html).
#   - Source the typical modules for IntelMPI.
#   - Add the newly installed libraries to your path, i.e run
#        `export PATH=/path-to-autotools/local/bin:$PATH`
#   - Run this script

CURR_DIR=$(pwd)
PACKAGE=p4est
INSTALL_ROOT=$MYSOFTWARE
P4EST_INSTALL=$INSTALL_ROOT/$PACKAGE/$P4EST_VERSION-$MPI_VERSION
ROOT_DIR=$INSTALL_ROOT/tmp
SOURCES_DIR=$ROOT_DIR/$PACKAGE-$P4EST_VERSION
BUILD_DIR=$SOURCES_DIR/build

git clone git@github.com:cburstedde/p4est.git $SOURCES_DIR
cd $SOURCES_DIR
# git checkout v2.2
git submodule init && git submodule update
./bootstrap
./configure --prefix=$P4EST_INSTALL $CONFIGURE_FLAGS --without-blas --without-lapack --enable-mpi --disable-dependency-tracking
make
make install
rm -rf $ROOT_DIR/$TAR_FILE $SOURCES_DIR
cd $CURR_DIR
