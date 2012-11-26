/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */

#include "ncx.h"

/* ftype is the variable's nc_type defined in file, eg. int64
 * btype is the I/O buffer's C data type, eg. long long
 * buftype is I/O bufer's MPI data type, eg. MPI_UNSIGNED_LONG_LONG
 * apitype is data type appeared in the API names, eg. ncmpi_get_vara_longlong
 */

/*---- x_int ----------------------------------------------------------------*/

#if SHORT_MAX == X_INT_MAX
typedef short ix_int;
#define SIZEOF_IX_INT SIZEOF_SHORT
#define IX_INT_MAX SHORT_MAX
#elif INT_MAX  >= X_INT_MAX
typedef int ix_int;
#define SIZEOF_IX_INT SIZEOF_INT
#define IX_INT_MAX INT_MAX
#elif LONG_MAX  >= X_INT_MAX
typedef long ix_int;
#define SIZEOF_IX_INT SIZEOF_LONG
#define IX_INT_MAX LONG_MAX
#else
#error "ix_int implementation"
#endif


static void
get_ix_int(const void *xp, ix_int *ip)
{
    const uchar *cp = (const uchar *) xp;

    *ip = *cp++ << 24;
#if SIZEOF_IX_INT > X_SIZEOF_INT
    if (*ip & 0x80000000)       /* extern is negative */
        *ip |= (~(0xffffffff)); /* N.B. Assumes "twos complement" */
#endif
    *ip |= (*cp++ << 16);
    *ip |= (*cp++ << 8);
    *ip |= *cp; 
}

static void
put_ix_int(void *xp, const ix_int *ip)
{
    uchar *cp = (uchar *) xp;

    *cp++ = (*ip) >> 24;
    *cp++ = ((*ip) & 0x00ff0000) >> 16;
    *cp++ = ((*ip) & 0x0000ff00) >>  8;
    *cp   = ((*ip) & 0x000000ff);
}

/*----< ncmpix_get_int_short() >---------------------------------------------*/
static int
ncmpix_get_int_short(const void *xp, short *ip)
{
#if SIZEOF_IX_INT == SIZEOF_SHORT && IX_INT_MAX == SHORT_MAX
    get_ix_int(xp, (ix_int *)ip);
    return NC_NOERR;
#else
    ix_int xx=0;
    get_ix_int(xp, &xx);
    *ip = xx;
#  if IX_INT_MAX > SHORT_MAX
    if (xx > SHORT_MAX || xx < SHORT_MIN)
        return NC_ERANGE;
#  endif
    return NC_NOERR;
#endif
}

/*----< ncmpix_get_int_ushort() >--------------------------------------------*/
static int
ncmpix_get_int_ushort(const void *xp, ushort *ip)
{
    ix_int xx=0;
    get_ix_int(xp, &xx);
    *ip = xx;
#if IX_INT_MAX > USHORT_MAX
    if (xx > USHORT_MAX || xx < 0)
        return NC_ERANGE;
#else
    if (xx < 0)
        return NC_ERANGE;
#endif
    return NC_NOERR;
}

#if SIZEOF_IX_INT != SIZEOF_INT
/*----< ncmpix_get_int_int() >-----------------------------------------------*/
static int
ncmpix_get_int_int(const void *xp, int *ip)
{
#if SIZEOF_IX_INT == SIZEOF_INT && IX_INT_MAX == INT_MAX
    get_ix_int(xp, (ix_int *)ip);
    return NC_NOERR;
#else
    ix_int xx;
    get_ix_int(xp, &xx);
    *ip = xx;
#  if IX_INT_MAX > INT_MAX
    if (xx > INT_MAX || xx < INT_MIN)
        return NC_ERANGE;
#  endif
    return NC_NOERR;
#endif
}
#endif

/*----< ncmpix_get_int_uint() >-----------------------------------------------*/
static int
ncmpix_get_int_uint(const void *xp, uint *ip)
{
    ix_int xx;
    get_ix_int(xp, &xx);
    *ip = xx;
#if IX_INT_MAX > UINT_MAX
    if (xx > UINT_MAX || xx < 0)
        return NC_ERANGE;
#else
    if (xx < 0)
        return NC_ERANGE;
#endif
    return NC_NOERR;
}

/*----< ncmpix_get_int_long() >----------------------------------------------*/
static int
ncmpix_get_int_long(const void *xp, long *ip)
{
#if SIZEOF_IX_INT == SIZEOF_LONG && IX_INT_MAX == LONG_MAX
    get_ix_int(xp, (ix_int *)ip);
    return NC_NOERR;
#else
    ix_int xx;
    get_ix_int(xp, &xx);
    *ip = xx;
#  if IX_INT_MAX > LONG_MAX    /* unlikely */
    if (xx > LONG_MAX || xx < LONG_MIN)
        return NC_ERANGE;
#  endif
    return NC_NOERR;
#endif
}


#define GET_INT(btype, range_check)                                           \
static int                                                                    \
ncmpix_get_int_##btype(const void *xp, btype *ip)                             \
{                                                                             \
    ix_int xx;                                                                \
    get_ix_int(xp, &xx);                                                      \
    *ip = xx;                                                                 \
    range_check      /* check if can fit into btype */                        \
    return NC_NOERR;                                                          \
}
/*----< ncmpix_get_int_schar() >---------------------------------------------*/
/*----< ncmpix_get_int_uchar() >---------------------------------------------*/
/*----< ncmpix_get_int_float() >---------------------------------------------*/
/*----< ncmpix_get_int_double() >--------------------------------------------*/
/*----< ncmpix_get_int_int64() >---------------------------------------------*/
GET_INT(schar, if (xx > SCHAR_MAX || xx < SCHAR_MIN) return NC_ERANGE;)
GET_INT(uchar, if (xx > UCHAR_MAX || xx < 0)         return NC_ERANGE;)
GET_INT(float,)
GET_INT(double,)
GET_INT(int64,)

/*----< ncmpix_get_int_uint64() >--------------------------------------------*/
static int
ncmpix_get_int_uint64(const void *xp, uint64 *ip)
{
    ix_int xx;
    get_ix_int(xp, &xx);
    *ip = xx;
    if (xx < 0)
        return NC_ERANGE;
    return NC_NOERR;
}


/*----< ncmpix_put_int_schar() >---------------------------------------------*/
static int
ncmpix_put_int_schar(void *xp, const schar *ip)
{
    uchar *cp = (uchar *) xp;
    if (*ip & 0x80) {
        *cp++ = 0xff;
        *cp++ = 0xff;
        *cp++ = 0xff;
    }
    else {
        *cp++ = 0x00;
        *cp++ = 0x00;
        *cp++ = 0x00;
    }
    *cp = (uchar)*ip;
    return NC_NOERR;
}

/*----< ncmpix_put_int_uchar() >---------------------------------------------*/
static int
ncmpix_put_int_uchar(void *xp, const uchar *ip)
{
    uchar *cp = (uchar *) xp;
    *cp++ = 0x00;
    *cp++ = 0x00;
    *cp++ = 0x00;
    *cp   = *ip;
    return NC_NOERR;
}

/*----< ncmpix_put_int_short() >---------------------------------------------*/
static int
ncmpix_put_int_short(void *xp, const short *ip)
{
#if SIZEOF_IX_INT == SIZEOF_SHORT && IX_INT_MAX == SHORT_MAX
    put_ix_int(xp, (ix_int *)ip);
    return NC_NOERR;
#else
    ix_int xx = (ix_int)(*ip);
    put_ix_int(xp, &xx);
#   if IX_INT_MAX < SHORT_MAX
    if(*ip > X_INT_MAX || *ip < X_INT_MIN)
        return NC_ERANGE;
#   endif
    return NC_NOERR;
#endif
}

/*----< ncmpix_put_int_ushort() >--------------------------------------------*/
static int
ncmpix_put_int_ushort(void *xp, const ushort *ip)
{
    ix_int xx = (ix_int)(*ip);
    put_ix_int(xp, &xx);
#if IX_INT_MAX < USHORT_MAX
    if (*ip > X_INT_MAX)
        return NC_ERANGE;
#endif
    return NC_NOERR;
}

#if SIZEOF_IX_INT != SIZEOF_INT
/*----< ncmpix_put_int_int() >-----------------------------------------------*/
static int
ncmpix_put_int_int(void *xp, const int *ip)
{
#if SIZEOF_IX_INT == SIZEOF_INT && IX_INT_MAX == INT_MAX
    put_ix_int(xp, (ix_int *)ip);
    return NC_NOERR;
#else
    ix_int xx = (ix_int)(*ip);
    put_ix_int(xp, &xx);
#   if IX_INT_MAX < INT_MAX
    if(*ip > X_INT_MAX || *ip < X_INT_MIN)
        return NC_ERANGE;
#   endif
    return NC_NOERR;
#endif
}
#endif

/*----< ncmpix_put_int_uint() >----------------------------------------------*/
static int
ncmpix_put_int_uint(void *xp, const uint *ip)
{
    ix_int xx = (ix_int)(*ip);
    put_ix_int(xp, &xx);
    if (*ip > X_INT_MAX)
        return NC_ERANGE;
    return NC_NOERR;
}

/*----< ncmpix_put_int_long() >----------------------------------------------*/
static int
ncmpix_put_int_long(void *xp, const long *ip)
{
#if SIZEOF_IX_INT == SIZEOF_LONG && IX_INT_MAX == LONG_MAX
    put_ix_int(xp, (ix_int *)ip);
    return NC_NOERR;
#else
    ix_int xx = (ix_int)(*ip);
    put_ix_int(xp, &xx);
#   if IX_INT_MAX < LONG_MAX
    if(*ip > X_INT_MAX || *ip < X_INT_MIN)
        return NC_ERANGE;
#   endif
    return NC_NOERR;
#endif
}

#define PUT_INT(btype)                                                        \
static int                                                                    \
ncmpix_put_int_##btype(void *xp, const btype *ip)                             \
{                                                                             \
    ix_int xx = (ix_int)(*ip);                                                \
    put_ix_int(xp, &xx);                                                      \
    /* typecast the MAX/MIN to double, otherwise it can fail                  \
     * example, when *ip == 2147483648.0 and X_INT_MAX==2147483647            \
     * (float)X_INT_MAX will become 2147483648.0                              \
     */                                                                       \
    if (*ip > (double)X_INT_MAX || *ip < (double)X_INT_MIN)                   \
        return NC_ERANGE;                                                     \
    return NC_NOERR;                                                          \
}

/*----< ncmpix_put_int_float() >---------------------------------------------*/
/*----< ncmpix_put_int_double() >--------------------------------------------*/
/*----< ncmpix_put_int_int64() >---------------------------------------------*/
PUT_INT(float)
PUT_INT(double)
PUT_INT(int64)

/*----< ncmpix_put_int_uint64() >--------------------------------------------*/
static int
ncmpix_put_int_uint64(void *xp, const uint64 *ip)
{
    ix_int xx = (ix_int)(*ip);
    put_ix_int(xp, &xx);
    /* typecast the MAX/MIN to double, otherwise it can fail
     * example, when *ip == 2147483648.0 and X_INT_MAX==2147483647
     * (float)X_INT_MAX will become 2147483648.0
     */
    if (*ip > (double)X_INT_MAX)
        return NC_ERANGE;
    return NC_NOERR;
}


 

/*---- int ------------------------------------------------------------------*/

#define GETN_INT(btype)                                                       \
int                                                                           \
ncmpix_getn_int_##btype(const void **xpp, MPI_Offset nelems, btype *tp)       \
{                                                                             \
    const char *xp = (const char *) *xpp;                                     \
    int status = NC_NOERR;                                                    \
                                                                              \
    for ( ; nelems != 0; nelems--, xp += X_SIZEOF_INT, tp++) {                \
        const int lstatus = ncmpix_get_int_##btype(xp, tp);                   \
        if (lstatus != NC_NOERR) status = lstatus;                            \
    }                                                                         \
                                                                              \
    *xpp = (void *)xp;                                                        \
    return status;                                                            \
}

/*----< ncmpix_getn_int_schar() >--------------------------------------------*/
/*----< ncmpix_getn_int_uchar() >--------------------------------------------*/
/*----< ncmpix_getn_int_short() >--------------------------------------------*/
/*----< ncmpix_getn_int_float() >--------------------------------------------*/
/*----< ncmpix_getn_int_double() >-------------------------------------------*/
/*----< ncmpix_getn_int_ushort() >-------------------------------------------*/
/*----< ncmpix_getn_int_uint() >---------------------------------------------*/
/*----< ncmpix_getn_int_int64() >--------------------------------------------*/
/*----< ncmpix_getn_int_uint64() >-------------------------------------------*/
GETN_INT(schar)
GETN_INT(uchar)
GETN_INT(short)
GETN_INT(float)
GETN_INT(double)
GETN_INT(ushort)
GETN_INT(uint)
GETN_INT(int64)
GETN_INT(uint64)

/*----< ncmpix_getn_int_int() >----------------------------------------------*/
#if X_SIZEOF_INT == SIZEOF_INT
/* optimized version */
int
ncmpix_getn_int_int(const void **xpp, MPI_Offset nelems, int *tp)
{
# ifdef WORDS_BIGENDIAN
    memcpy(tp, *xpp, nelems * sizeof(int));
# else
    ncmpii_swapn4b(tp, *xpp, nelems);
# endif
    *xpp = (const void *)((const char *)(*xpp) + nelems * X_SIZEOF_INT);
    return NC_NOERR;
}
#else
GETN_INT(int)
#endif

/*----< ncmpix_getn_int_long() >---------------------------------------------*/
#if X_SIZEOF_INT == SIZEOF_LONG
/* optimized version */
int
ncmpix_getn_int_long(const void **xpp, MPI_Offset nelems, long *tp)
{
# ifdef WORDS_BIGENDIAN
    memcpy(tp, *xpp, nelems * sizeof(long));
# else
    ncmpii_swapn4b(tp, *xpp, nelems);
# endif
    *xpp = (const void *)((const char *)(*xpp) + nelems * X_SIZEOF_INT);
    return NC_NOERR;
}
#else
GETN_INT(long)
#endif


#define PUTN_INT(btype)                                                       \
int                                                                           \
ncmpix_putn_int_##btype(void **xpp, MPI_Offset nelems, const btype *tp)       \
{                                                                             \
    char *xp = (char *) *xpp;                                                 \
    int status = NC_NOERR;                                                    \
                                                                              \
    for ( ; nelems != 0; nelems--, xp += X_SIZEOF_INT, tp++) {                \
        int lstatus = ncmpix_put_int_##btype(xp, tp);                         \
        if (lstatus != NC_NOERR) status = lstatus;                            \
    }                                                                         \
                                                                              \
    *xpp = (void *)xp;                                                        \
    return status;                                                            \
}

/*----< ncmpix_putn_int_schar() >--------------------------------------------*/
/*----< ncmpix_putn_int_uchar() >--------------------------------------------*/
/*----< ncmpix_putn_int_short() >--------------------------------------------*/
/*----< ncmpix_putn_int_float() >--------------------------------------------*/
/*----< ncmpix_putn_int_double() >-------------------------------------------*/
/*----< ncmpix_putn_int_ushort() >-------------------------------------------*/
/*----< ncmpix_putn_int_uint() >---------------------------------------------*/
/*----< ncmpix_putn_int_int64() >--------------------------------------------*/
/*----< ncmpix_putn_int_uint64() >-------------------------------------------*/
PUTN_INT(schar)
PUTN_INT(uchar)
PUTN_INT(short)
PUTN_INT(float)
PUTN_INT(double)
PUTN_INT(ushort)
PUTN_INT(uint)
PUTN_INT(int64)
PUTN_INT(uint64)

/*----< ncmpix_putn_int_int() >----------------------------------------------*/
#if X_SIZEOF_INT == SIZEOF_INT
/* optimized version */
int
ncmpix_putn_int_int(void **xpp, MPI_Offset nelems, const int *tp)
{
# ifdef WORDS_BIGENDIAN
    memcpy(*xpp, tp, nelems * X_SIZEOF_INT);
# else
    ncmpii_swapn4b(*xpp, tp, nelems);
# endif
    *xpp = (void *)((char *)(*xpp) + nelems * X_SIZEOF_INT);
    return NC_NOERR;
}
#else
PUTN_INT(int)
#endif

/*----< ncmpix_putn_int_long() >---------------------------------------------*/
#if X_SIZEOF_INT == SIZEOF_LONG
/* optimized version */
int
ncmpix_putn_int_long(void **xpp, MPI_Offset nelems, const long *tp)
{
# ifdef WORDS_BIGENDIAN
    memcpy(*xpp, tp, nelems * X_SIZEOF_INT);
# else
    ncmpii_swapn4b(*xpp, tp, nelems);
# endif
    *xpp = (void *)((char *)(*xpp) + nelems * X_SIZEOF_INT);
    return NC_NOERR;
}
#else
PUTN_INT(long)
#endif

/*----< ncmpix_get_long_long() >---------------------------------------------*/
/* wkliao: long type is not supported anymore. NC_LONG == NC_INT now
           so, ncmpix_get_long_long is actually ncmpix_get_int_long()
	   and ncmpix_put_long_long is actually ncmpix_put_int_long().  We
	   removed the now-unused functions 
	   - 
 */


/* wkliao: long type is not supported anymore. NC_LONG == NC_INT now
           so, ncmpix_getn_long_long is actually ncmpix_getn_int_long()
           and ncmpix_putn_long_long is actually ncmpix_putn_int_long()
 */
int
ncmpix_getn_long_long(const void **xpp, MPI_Offset nelems, long *tp)
{
    return ncmpix_getn_int_long(xpp, nelems, tp);
}

int
ncmpix_putn_long_long(void **xpp, MPI_Offset nelems, const long *tp)
{
    return ncmpix_putn_int_long(xpp, nelems, tp);
}

