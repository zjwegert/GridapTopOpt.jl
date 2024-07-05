# Install PETSc
# Requires two environment variables to be setup:
#   - PETSC_VERSION   :: PETSc version
#   - MPI_VERSION     :: MPI version and library
#   - CONFIGURE_FLAGS :: --with-xxx style flags to indicate compilers

CURR_DIR=$(pwd)
PACKAGE=petsc
INSTALL_ROOT=$MYSOFTWARE
PETSC_INSTALL=$INSTALL_ROOT/$PACKAGE/$PETSC_VERSION-$MPI_VERSION
TAR_FILE=$PACKAGE-$PETSC_VERSION.tar.gz
URL="https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/"
ROOT_DIR=/tmp
SOURCES_DIR=$ROOT_DIR/$PACKAGE-$PETSC_VERSION
BUILD_DIR=$SOURCES_DIR/build
wget -q $URL/$TAR_FILE -O $ROOT_DIR/$TAR_FILE
mkdir -p $SOURCES_DIR
tar xzf $ROOT_DIR/$TAR_FILE -C $SOURCES_DIR --strip-components=1
cd $SOURCES_DIR
./configure --prefix=$PETSC_INSTALL $CONFIGURE_FLAGS \
    --download-mumps --download-scalapack --download-parmetis --download-metis \
    --download-ptscotch --with-debugging --with-x=0 --with-shared=1 \
    --with-mpi=1 --with-64-bit-indices
make
make install
rm -rf $ROOT_DIR/$TAR_FILE $SOURCES_DIR
cd $CURR_DIR
