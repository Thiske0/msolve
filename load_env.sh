source ~/spack/share/spack/setup-env.sh
spack load mpfr
export MPFR_ROOT=$(spack location -i mpfr)
export FLINT_ROOT=$(spack location -i flint)

# Tell configure where to find the library
./configure  ac_cv_func_malloc_0_nonnull=yes ac_cv_func_realloc_0_nonnull=yes \
    LDFLAGS="-L$MPFR_ROOT/lib -L$FLINT_ROOT/lib" CPPFLAGS="-I$MPFR_ROOT/include -I$FLINT_ROOT/include"

export LD_LIBRARY_PATH=$MPFR_ROOT/lib:$FLINT_ROOT/lib:$LD_LIBRARY_PATH