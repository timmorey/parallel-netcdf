/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */

#include "nc.h"
#include "ncx.h"
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <assert.h>

#include "ncmpidtype.h"
#include "macro.h"


/* ftype is the variable's nc_type defined in file, eg. int64
 * btype is the I/O buffer's C data type, eg. long long
 * buftype is I/O bufer's MPI data type, eg. MPI_UNSIGNED_LONG_LONG
 * apitype is data type appeared in the API names, eg. ncmpi_get_vara_longlong
 */

/*----< ncmpi_bput_var() >---------------------------------------------------*/
int
ncmpi_bput_var(int           ncid,
               int           varid,
               const void   *buf,
               MPI_Offset    bufcount,
               MPI_Datatype  buftype,
               int          *reqid)
{
    int         status;
    NC         *ncp;
    NC_var     *varp;
    MPI_Offset *start, *count;

    *reqid = NC_REQ_NULL;
    CHECK_NCID
    if (ncp->abuf == NULL) return NC_ENULLABUF;
    CHECK_WRITE_PERMISSION
    if (NC_indef(ncp)) return NC_EINDEFINE;
    CHECK_VARID(varid, varp)
    GET_FULL_DIMENSIONS

    /* bput_var is a special case of bput_vara */
    status = ncmpii_igetput_varm(ncp, varp, start, count, NULL, NULL,
                                 (void*)buf, bufcount, buftype, reqid,
                                 WRITE_REQ, 1);
    if (varp->ndims > 0) NCI_Free(start);

    return status;
}


#define BPUT_VAR_TYPE(apitype, btype, buftype)                          \
int                                                                     \
ncmpi_bput_var_##apitype(int          ncid,                             \
                         int          varid,                            \
                         const btype *op,                               \
                         int         *reqid)                            \
{                                                                       \
    int         status;                                                 \
    NC         *ncp;                                                    \
    NC_var     *varp;                                                   \
    MPI_Offset  nelems, *start, *count;                                 \
                                                                        \
    *reqid = NC_REQ_NULL;                                               \
    CHECK_NCID                                                          \
    if (ncp->abuf == NULL) return NC_ENULLABUF;                         \
    CHECK_WRITE_PERMISSION                                              \
    if (NC_indef(ncp)) return NC_EINDEFINE;                             \
    CHECK_VARID(varid, varp)                                            \
    GET_TOTAL_NUM_ELEMENTS                                              \
    GET_FULL_DIMENSIONS                                                 \
                                                                        \
    /* bput_var is a special case of bput_varm */                       \
    status = ncmpii_igetput_varm(ncp, varp, start, count, NULL, NULL,   \
                                 (void*)op, nelems, buftype, reqid,     \
                                 WRITE_REQ, 1);                         \
    if (varp->ndims > 0) NCI_Free(start);                               \
                                                                        \
    return status;                                                      \
}

/*----< ncmpi_bput_var_text() >-----------------------------------------------*/
/*----< ncmpi_bput_var_schar() >----------------------------------------------*/
/*----< ncmpi_bput_var_uchar() >----------------------------------------------*/
/*----< ncmpi_bput_var_short() >----------------------------------------------*/
/*----< ncmpi_bput_var_ushort() >---------------------------------------------*/
/*----< ncmpi_bput_var_int() >------------------------------------------------*/
/*----< ncmpi_bput_var_uint() >-----------------------------------------------*/
/*----< ncmpi_bput_var_long() >-----------------------------------------------*/
/*----< ncmpi_bput_var_float() >----------------------------------------------*/
/*----< ncmpi_bput_var_double() >---------------------------------------------*/
/*----< ncmpi_bput_var_longlong() >-------------------------------------------*/
/*----< ncmpi_bput_var_ulonglong() >------------------------------------------*/
BPUT_VAR_TYPE(text,      char,               MPI_CHAR)
BPUT_VAR_TYPE(schar,     schar,              MPI_BYTE)
BPUT_VAR_TYPE(uchar,     uchar,              MPI_UNSIGNED_CHAR)
BPUT_VAR_TYPE(short,     short,              MPI_SHORT)
BPUT_VAR_TYPE(ushort,    ushort,             MPI_UNSIGNED_SHORT)
BPUT_VAR_TYPE(int,       int,                MPI_INT)
BPUT_VAR_TYPE(uint,      uint,               MPI_UNSIGNED)
BPUT_VAR_TYPE(long,      long,               MPI_LONG)
BPUT_VAR_TYPE(float,     float,              MPI_FLOAT)
BPUT_VAR_TYPE(double,    double,             MPI_DOUBLE)
BPUT_VAR_TYPE(longlong,  long long,          MPI_LONG_LONG_INT)
BPUT_VAR_TYPE(ulonglong, unsigned long long, MPI_UNSIGNED_LONG_LONG)
// BPUT_VAR_TYPE(string, char*,              MPI_CHAR)
/* string is not yet supported */


