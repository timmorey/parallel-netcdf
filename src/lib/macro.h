/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */

#ifndef _H_MACRO
#define _H_MACRO

#define WRITE_REQ 0
#define READ_REQ  1

#define INDEP_IO 0
#define COLL_IO  1

#ifndef MAX
#define MAX(mm,nn) (((mm) > (nn)) ? (mm) : (nn))
#endif
#ifndef MIN
#define MIN(mm,nn) (((mm) < (nn)) ? (mm) : (nn))
#endif

void *NCI_Malloc_fn(size_t size, int lineno, const char *fname);
void *NCI_Calloc_fn(size_t nelem, size_t elsize, int lineno, const char *fname);
void *NCI_Realloc_fn(void *ptr, size_t size, int lineno, const char *fname);
void  NCI_Free_fn(void *ptr, int lineno, const char *fname);

#define NCI_Malloc(a)    NCI_Malloc_fn(a,__LINE__,__FILE__)
#define NCI_Calloc(a,b)  NCI_Calloc_fn(a,b,__LINE__,__FILE__)
#define NCI_Realloc(a,b) NCI_Realloc_fn(a,b,__LINE__,__FILE__)
#define NCI_Free(a)      NCI_Free_fn(a,__LINE__,__FILE__)


#define CHECK_MPI_ERROR(str, err) {                                           \
    if (mpireturn != MPI_SUCCESS) {                                           \
        char errorString[MPI_MAX_ERROR_STRING];                               \
        int rank, errorStringLen;                                             \
        MPI_Comm_rank(ncp->nciop->comm, &rank);                               \
        MPI_Error_string(mpireturn, errorString, &errorStringLen);            \
        printf("%2d: MPI Failure at line %d of %s (%s : %s)\n",               \
               rank, __LINE__, __FILE__, str, errorString);                   \
        return err;                                                           \
    }                                                                         \
}

/* API error will terminate the API call, not the entire program */
#define CHECK_NCID {                                                          \
    status = ncmpii_NC_check_id(ncid, &ncp);                                  \
    if (status != NC_NOERR) { /* API error */                                 \
        /* uncomment to print debug message                                   \
        int rank;                                                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
        printf("%2d: Invalid ncid(%d) at line %d of %s\n",                    \
               rank, ncid, __LINE__, __FILE__);                               \
        */                                                                    \
        return status; /* abort the API now */                                \
    }                                                                         \
}

#define CHECK_WRITE_PERMISSION {                                              \
    if (NC_readonly(ncp)) { /* API error */                                   \
        int rank;                                                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
        printf("%2d: No file write permission at line %d of %s\n",            \
               rank, __LINE__, __FILE__);                                     \
        return NC_EPERM;                                                      \
    }                                                                         \
}

#define CHECK_INDEP_FH {                                                      \
    /* check to see that the independent MPI file handle is opened */         \
    status =                                                                  \
    ncmpii_check_mpifh(ncp, &(ncp->nciop->independent_fh), MPI_COMM_SELF, 0); \
    if (status != NC_NOERR) {                                                 \
        /* uncomment to print debug message                                   \
        int rank;                                                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
        printf("%2d: Error - MPI indep file handle at line %d of %s\n",       \
               rank, __LINE__, __FILE__);                                     \
        */                                                                    \
        return status;  /* abort the API now */                               \
    }                                                                         \
}

#define CHECK_COLLECTIVE_FH {                                                 \
    /* check to see that the collective MPI file handle is opened */          \
    status =                                                                  \
    ncmpii_check_mpifh(ncp, &(ncp->nciop->collective_fh),ncp->nciop->comm,1); \
    if (status != NC_NOERR) {                                                 \
        /* uncomment to print debug message                                   \
        int rank;                                                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
        printf("%2d: Error - MPI collective file handle at line %d of %s\n",  \
               rank, __LINE__, __FILE__);                                     \
        */                                                                    \
        return status;  /* abort the API now */                               \
    }                                                                         \
} 

#define CHECK_VARID(varid, varp) {                                            \
    varp = ncmpii_NC_lookupvar(ncp, varid);                                   \
    if (varp == NULL) { /* API error */                                       \
        /* uncomment to print debug message                                   \
        int rank;                                                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
        printf("%2d: Invalid varid(%d) at line %d of %s\n",                   \
               rank, varid, __LINE__, __FILE__);                              \
        */                                                                    \
        return NC_ENOTVAR;                                                    \
    }                                                                         \
}

#define GET_ONE_COUNT {                                                       \
    int _i;                                                                   \
    count = (MPI_Offset*) NCI_Malloc(varp->ndims * sizeof(MPI_Offset));       \
    for (_i=0; _i<varp->ndims; _i++)                                          \
        count[_i] = 1;                                                        \
}

#define GET_TOTAL_NUM_ELEMENTS {                                              \
    int ndims = varp->ndims;                                                  \
    if (ndims == 0)                                                           \
        nelems = 1;                                                           \
    else if (!IS_RECVAR(varp))                                                \
        nelems = varp->dsizes[0];                                             \
    else if (ndims > 1)                                                       \
        nelems = ncp->numrecs * varp->dsizes[1];                              \
    else                                                                      \
        nelems = ncp->numrecs;                                                \
}

#define GET_NUM_ELEMENTS {                                                    \
    int _i;                                                                   \
    nelems = 1;                                                               \
    for (_i=0; _i<varp->ndims; _i++)                                          \
        nelems *= count[_i];                                                  \
}

#define GET_FULL_DIMENSIONS {                                                 \
    int _i;                                                                   \
    start = (MPI_Offset*) NCI_Malloc(2 * varp->ndims * sizeof(MPI_Offset));   \
    count = start + varp->ndims;                                              \
                                                                              \
    for (_i=0; _i<varp->ndims; _i++) {                                        \
        NC_dim *dimp;                                                         \
        dimp = ncmpii_elem_NC_dimarray(&ncp->dims, (size_t)varp->dimids[_i]); \
        if (dimp->size == NC_UNLIMITED)                                       \
            count[_i] = NC_get_numrecs(ncp);                                  \
        else                                                                  \
            count[_i] = dimp->size;                                           \
        start[_i] = 0;                                                        \
    }                                                                         \
}

/*
  ncmpii_dtype_decode - Decode the MPI derived datatype to get the bottom
  level primitive datatype information.

  Input:
. buftype - The MPI derived data type to be decoded (can be predefined type).

  Output:
. ptype - The bottom level MPI primitive datatype (only one allowed) in buftype
. el_size - The element size in bytes of the ptype
. nelems - Number of elements/entries of such ptype in one buftype object
. buftype_is_contig - Whether buftype is a contiguous number of ptype
*/
#define CHECK_DATATYPE(buftype, ptype, esize, nelems, buftype_is_contig) {    \
    int isderived;                                                            \
    err = ncmpii_dtype_decode(buftype, &(ptype), &(esize), &(nelems),         \
                              &isderived, &buftype_is_contig);                \
    if (err != NC_NOERR) { /* API error */                                    \
        /* uncomment to print debug message                                   \
        int rank;                                                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
        printf("%2d: buftype decode error at line %d of %s\n",                \
               rank, __LINE__, __FILE__);                                     \
        */                                                                    \
        goto err_check;                                                       \
        /* cannot return now, for collective call must return collectively */ \
    }                                                                         \
}

#define CHECK_ECHAR(varp) {                                                   \
    /* netcdf makes it illegal to type convert char and numbers */            \
    if ( ncmpii_echar((varp)->type, ptype) ) { /* API error */                \
        /* uncomment to print debug message                                   \
        int rank;                                                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
        printf("%2d: buftype cannot convert to CHAR at line %d of %s\n",      \
               rank, __LINE__, __FILE__);                                     \
        */                                                                    \
        err = NC_ECHAR;                                                       \
        goto err_check;                                                       \
        /* cannot return now, for collective call must return collectively */ \
    }                                                                         \
}

#define CHECK_NELEMS(varp, fnelems, fcount, bnelems, bufcount, nbytes) {      \
    int _i;                                                                   \
    /* bnelems is the number of elements in the I/O buffer and the element    \
     * is of MPI primitive type. When input, bnelems is the number of         \
     * MPI primitive type elements in the user's derived data type.           \
     * Here, we make it the total MPI primitive type elements in the          \
     * user's buffer                                                          \
     */                                                                       \
    bnelems *= bufcount;                                                      \
                                                                              \
    /* fnelems is the total number of nc_type elements calculated from        \
     * fcount[]. fcount[] is the access count[ to the variable defined in     \
     * the netCDF file.                                                       \
     */                                                                       \
    fnelems = 1;                                                              \
    for (_i=0; _i<(varp)->ndims; _i++) {                                      \
        if (fcount[_i] < 0) { /* API error */                                 \
            err = NC_ENEGATIVECNT;                                            \
            goto err_check;                                                   \
            /* for collective call must return collectively */                \
        }                                                                     \
        fnelems *= fcount[_i];                                                \
    }                                                                         \
                                                                              \
    /* check mismatch between bnelems and fnelems */                          \
    if (fnelems != bnelems) {                                                 \
        if (warning == NC_NOERR)                                              \
            warning = NC_EIOMISMATCH;                                         \
        (fnelems>bnelems) ? (fnelems=bnelems) : (bnelems=fnelems);            \
        /* only handle partial of the request, smaller number of the two */   \
    }                                                                         \
    /* now fnelems == bnelems */                                              \
                                                                              \
    /* nbytes is the amount of this request in bytes */                       \
    nbytes = fnelems * (varp)->xsz;                                           \
    if (nbytes < 0) { /* API error */                                         \
        /* uncomment to print debug message                                   \
        int rank;                                                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                 \
        printf("%2d: Error - negative request amount at line %d of %s\n",     \
               rank, __LINE__, __FILE__);                                     \
        */                                                                    \
        err = NC_ENEGATIVECNT;                                                \
        goto err_check;                                                       \
        /* cannot return now, for collective call must return collectively */ \
    }                                                                         \
}

#define DATATYPE_GET_CONVERT(vartype, inbuf, outbuf, cnelems, memtype) {      \
    /* vartype is the variable's data type defined in the nc file             \
     * memtype is the I/O buffers data type (MPI_Datatype)  */                \
    switch(vartype) {                                                         \
        case NC_BYTE:                                                         \
            status = ncmpii_x_getn_schar(inbuf, outbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_UBYTE:                                                        \
            status = ncmpii_x_getn_uchar(inbuf, outbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_SHORT:                                                        \
            status = ncmpii_x_getn_short(inbuf, outbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_USHORT:                                                       \
            status = ncmpii_x_getn_ushort(inbuf, outbuf, cnelems, memtype);   \
            break;                                                            \
        case NC_INT:                                                          \
            status = ncmpii_x_getn_int(inbuf, outbuf, cnelems, memtype);      \
            break;                                                            \
        case NC_UINT:                                                         \
            status = ncmpii_x_getn_uint(inbuf, outbuf, cnelems, memtype);     \
            break;                                                            \
        case NC_FLOAT:                                                        \
            status = ncmpii_x_getn_float(inbuf, outbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_DOUBLE:                                                       \
            status = ncmpii_x_getn_double(inbuf, outbuf, cnelems, memtype);   \
            break;                                                            \
        case NC_INT64:                                                        \
            status = ncmpii_x_getn_int64(inbuf, outbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_UINT64:                                                       \
            status = ncmpii_x_getn_uint64(inbuf, outbuf, cnelems, memtype);   \
            break;                                                            \
        default:                                                              \
            break;                                                            \
    }                                                                         \
}

#define DATATYPE_PUT_CONVERT(vartype, outbuf, inbuf, cnelems, memtype) {      \
    /* vartype is the variable's data type defined in the nc file             \
     * memtype is the I/O buffers data type (MPI_Datatype)  */                \
    switch(vartype) {                                                         \
        case NC_BYTE:                                                         \
            status = ncmpii_x_putn_schar(outbuf, inbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_UBYTE:                                                        \
            status = ncmpii_x_putn_uchar(outbuf, inbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_SHORT:                                                        \
            status = ncmpii_x_putn_short(outbuf, inbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_USHORT:                                                       \
            status = ncmpii_x_putn_ushort(outbuf, inbuf, cnelems, memtype);   \
            break;                                                            \
        case NC_INT:                                                          \
            status = ncmpii_x_putn_int(outbuf, inbuf, cnelems, memtype);      \
            break;                                                            \
        case NC_UINT:                                                         \
            status = ncmpii_x_putn_uint(outbuf, inbuf, cnelems, memtype);     \
            break;                                                            \
        case NC_FLOAT:                                                        \
            status = ncmpii_x_putn_float(outbuf, inbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_DOUBLE:                                                       \
            status = ncmpii_x_putn_double(outbuf, inbuf, cnelems, memtype);   \
            break;                                                            \
        case NC_INT64:                                                        \
            status = ncmpii_x_putn_int64(outbuf, inbuf, cnelems, memtype);    \
            break;                                                            \
        case NC_UINT64:                                                       \
            status = ncmpii_x_putn_uint64(outbuf, inbuf, cnelems, memtype);   \
            break;                                                            \
        default:                                                              \
            break;                                                            \
    }                                                                         \
}

#endif
