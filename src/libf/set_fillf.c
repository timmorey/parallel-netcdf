/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*  
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 * This file is automatically generated by buildiface -infile=../lib/pnetcdf.h -deffile=defs
 * DO NOT EDIT
 */
#include "mpinetcdf_impl.h"


#ifdef F77_NAME_UPPER
#define nfmpi_set_fill_ NFMPI_SET_FILL
#elif defined(F77_NAME_LOWER_2USCORE)
#define nfmpi_set_fill_ nfmpi_set_fill__
#elif !defined(F77_NAME_LOWER_USCORE)
#define nfmpi_set_fill_ nfmpi_set_fill
/* Else leave name alone */
#endif


/* Prototypes for the Fortran interfaces */
#include "mpifnetcdf.h"
FORTRAN_API int FORT_CALL nfmpi_set_fill_ ( int *v1, int *v2, MPI_Fint *v3 ){
    int ierr;
    ierr = ncmpi_set_fill( *v1, *v2, v3 );
    return ierr;
}
