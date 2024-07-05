# Install p4est from sources
# Requires two environment variables to be setup: 
#   - P4EST_VERSION   :: P4est version
#   - MPI_VERSION     :: MPI version and library
#   - CONFIGURE_FLAGS :: --with-xxx style flags for compiler configuration

CURR_DIR=$(pwd)
PACKAGE=p4est
INSTALL_ROOT=$HOME/bin
P4EST_INSTALL=$INSTALL_ROOT/$PACKAGE/$P4EST_VERSION-$MPI_VERSION
TAR_FILE=$PACKAGE-$P4EST_VERSION.tar.gz
URL="https://github.com/p4est/p4est.github.io/raw/master/release"
ROOT_DIR=/tmp
SOURCES_DIR=$ROOT_DIR/$PACKAGE-$P4EST_VERSION
BUILD_DIR=$SOURCES_DIR/build
wget -q $URL/$TAR_FILE -O $ROOT_DIR/$TAR_FILE
mkdir -p $SOURCES_DIR
tar xzf $ROOT_DIR/$TAR_FILE -C $SOURCES_DIR --strip-components=1
cd $SOURCES_DIR
./configure --prefix=$P4EST_INSTALL $CONFIGURE_FLAGS --without-blas --without-lapack --enable-mpi --disable-dependency-tracking
make --quiet
make --quiet install
rm -rf $ROOT_DIR/$TAR_FILE $SOURCES_DIR
cd $CURR_DIR
