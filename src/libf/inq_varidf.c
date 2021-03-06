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
#define nfmpi_inq_varid_ NFMPI_INQ_VARID
#elif defined(F77_NAME_LOWER_2USCORE)
#define nfmpi_inq_varid_ nfmpi_inq_varid__
#elif !defined(F77_NAME_LOWER_USCORE)
#define nfmpi_inq_varid_ nfmpi_inq_varid
/* Else leave name alone */
#endif


/* Prototypes for the Fortran interfaces */
#include "mpifnetcdf.h"
FORTRAN_API int FORT_CALL nfmpi_inq_varid_ ( int *v1, char *v2 FORT_MIXED_LEN(d2), MPI_Fint *v3 FORT_END_LEN(d2) ){
    int ierr;
    char *p2;

    {char *p = v2 + d2 - 1;
     int  li;
        while (*p == ' ' && p > v2) p--;
        p++;
        p2 = (char *)malloc( p-v2 + 1 );
        for (li=0; li<(p-v2); li++) { p2[li] = v2[li]; }
        p2[li] = 0; 
    }
    ierr = ncmpi_inq_varid( *v1, p2, v3 );
    free( p2 );

    if (!ierr) *v3 = *v3 + 1;
    return ierr;
}
