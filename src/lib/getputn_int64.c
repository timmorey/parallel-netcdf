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

/*---- x_int64 --------------------------------------------------------------*/

#define SWAP8B(x) {                    \
    uchar _c, *_p=(uchar*)(x);         \
    _c=_p[0]; _p[0]=_p[7]; _p[7]=_c;   \
    _c=_p[1]; _p[1]=_p[6]; _p[6]=_c;   \
    _c=_p[2]; _p[2]=_p[5]; _p[5]=_c;   \
    _c=_p[3]; _p[3]=_p[4]; _p[4]=_c;   \
}
/*
static void
get_ix_int64(const void *xp, int64 *ip)
{
    *ip = *((int64*)xp);
#ifndef WORDS_BIGENDIAN
    SWAP8B(ip);
#endif
}

static void
put_ix_int64(void *xp, const int64 *ip)
{
    *((int64*) xp) = *ip;
#ifndef WORDS_BIGENDIAN
    SWAP8B(xp);
#endif
}
*/

static void
get_ix_int64(const void *xp, int64 *ip)
{
    /* are these bit shifting faster than byte swap? */
    const uchar *cp = (const uchar *) xp;

    *ip  = ((int64)(*cp++) << 56);
    *ip |= ((int64)(*cp++) << 48);
    *ip |= ((int64)(*cp++) << 40);
    *ip |= ((int64)(*cp++) << 32);
    *ip |= ((int64)(*cp++) << 24);
    *ip |= ((int64)(*cp++) << 16);
    *ip |= ((int64)(*cp++) <<  8);
    *ip |=  (int64)*cp;
}

static void
put_ix_int64(void *xp, const int64 *ip)
{
    uchar *cp = (uchar *) xp;

    *cp++ = (*ip) >> 56;
    *cp++ = ((*ip) & 0x00ff000000000000LL) >> 48;
    *cp++ = ((*ip) & 0x0000ff0000000000LL) >> 40;
    *cp++ = ((*ip) & 0x000000ff00000000LL) >> 32;
    *cp++ = ((*ip) & 0x00000000ff000000LL) >> 24;
    *cp++ = ((*ip) & 0x0000000000ff0000LL) >> 16;
    *cp++ = ((*ip) & 0x000000000000ff00LL) >>  8;
    *cp   = ((*ip) & 0x00000000000000ffLL);
}

#define GET_INT64(btype, range_check)                                         \
static int                                                                    \
ncmpix_get_int64_##btype(const void *xp, btype *ip)                           \
{                                                                             \
    int64 xx;                                                                 \
    get_ix_int64(xp, &xx);                                                    \
    *ip = xx;                                                                 \
    range_check         /* check if can fit into btype */                     \
    return NC_NOERR;                                                          \
}
/* for smaller-sized   signed types, check if the got int64 is too big or too small (schar, short, int, long, float)
 * for smaller-sized unsigned types, check if the got int64 is too big or negative (uchar, ushort, uint)
 * for equal-sized     signed types, no check is needed (int64, double)
 * for equal-sized   unsigned types, check if the got int64 is negative (uint64)
 */
/*----< ncmpix_get_int64_schar() >-------------------------------------------*/
/*----< ncmpix_get_int64_short() >-------------------------------------------*/
/*----< ncmpix_get_int64_int() >---------------------------------------------*/
/*----< ncmpix_get_int64_long() >--------------------------------------------*/
/*----< ncmpix_get_int64_float() >-------------------------------------------*/
GET_INT64(schar,  if (xx > SCHAR_MAX || xx < SCHAR_MIN) return NC_ERANGE;)
GET_INT64(short,  if (xx > SHRT_MAX  || xx < SHRT_MIN)  return NC_ERANGE;)
GET_INT64(int,    if (xx > INT_MAX   || xx < INT_MIN)   return NC_ERANGE;)
#if SIZEOF_LONG == X_SIZEOF_INT
static int 
ncmpix_get_int64_long(const void *xp, long *ip) 
{                                              
    return ncmpix_get_int64_int(xp, (int*)ip);
}
#else
GET_INT64(long,   if (xx > LONG_MAX  || xx < LONG_MIN)  return NC_ERANGE;)
#endif
GET_INT64(float,  if (xx > FLT_MAX   || xx < -FLT_MAX)  return NC_ERANGE;)
/*----< ncmpix_get_int64_uchar() >-------------------------------------------*/
/*----< ncmpix_get_int64_ushort() >------------------------------------------*/
/*----< ncmpix_get_int64_uint() >--------------------------------------------*/
GET_INT64(uchar,  if (xx > UCHAR_MAX || xx < 0) return NC_ERANGE;)
GET_INT64(ushort, if (xx > USHRT_MAX || xx < 0) return NC_ERANGE;)
GET_INT64(uint,   if (xx > UINT_MAX  || xx < 0) return NC_ERANGE;)
/*----< ncmpix_get_int64_double() >------------------------------------------*/
/*----< ncmpix_get_int64_int64() >-------------------------------------------*/
GET_INT64(double,)
GET_INT64(int64,)
/*----< ncmpix_get_int64_uint64() >------------------------------------------*/
GET_INT64(uint64, if (xx <  0) return NC_ERANGE;)


#define PUT_INT64(btype, range_check)                                         \
static int                                                                    \
ncmpix_put_int64_##btype(void *xp, const btype *ip)                           \
{                                                                             \
    int64 xx = (int64) *ip;                                                   \
    put_ix_int64(xp, &xx);                                                    \
    range_check         /* check if can fit into int64 */                     \
    return NC_NOERR;                                                          \
}
/* for smaller-sized   signed types, no check is needed (schar, short, int, long, float)
 * for smaller-sized unsigned types, no check is needed (uchar, ushort, uint)
 * for equal-sized     signed types, no check is needed (int64, double)
 * for equal-sized   unsigned types, check if the put value is too big (uint64)
 */
/*----< ncmpix_put_int64_schar() >-------------------------------------------*/
/*----< ncmpix_put_int64_uchar() >-------------------------------------------*/
/*----< ncmpix_put_int64_short() >-------------------------------------------*/
/*----< ncmpix_put_int64_int() >---------------------------------------------*/
/*----< ncmpix_put_int64_long() >--------------------------------------------*/
/*----< ncmpix_put_int64_ushort() >------------------------------------------*/
/*----< ncmpix_put_int64_uint() >--------------------------------------------*/
/*----< ncmpix_put_int64_int64() >-------------------------------------------*/
PUT_INT64(schar,)
PUT_INT64(uchar,)
PUT_INT64(short,)
PUT_INT64(int,)
PUT_INT64(long,)
PUT_INT64(uint,)
PUT_INT64(ushort,)
PUT_INT64(int64,)
/*----< ncmpix_put_int64_float() >-------------------------------------------*/
/*----< ncmpix_put_int64_double() >------------------------------------------*/
PUT_INT64(float,  if (*ip > X_INT64_MAX || *ip < X_INT64_MIN) return NC_ERANGE;)
PUT_INT64(double, if (*ip > X_INT64_MAX || *ip < X_INT64_MIN) return NC_ERANGE;)
/*----< ncmpix_put_int64_uint64() >------------------------------------------*/
PUT_INT64(uint64, if (*ip > X_INT64_MAX) return NC_ERANGE;)


/*---- int64 ----------------------------------------------------------------*/

#define GETN_INT64(btype)                                                     \
int                                                                           \
ncmpix_getn_int64_##btype(const void **xpp, MPI_Offset nelems, btype *tp)     \
{                                                                             \
    const char *xp = (const char *) *xpp;                                     \
    int status = NC_NOERR;                                                    \
                                                                              \
    for ( ; nelems != 0; nelems--, xp += X_SIZEOF_INT64, tp++) {              \
        const int lstatus = ncmpix_get_int64_##btype(xp, tp);                 \
        if (lstatus != NC_NOERR) status = lstatus;                            \
    }                                                                         \
                                                                              \
    *xpp = (void *)xp;                                                        \
    return status;                                                            \
}
/*----< ncmpix_getn_int64_schar() >------------------------------------------*/
/*----< ncmpix_getn_int64_uchar() >------------------------------------------*/
/*----< ncmpix_getn_int64_short() >------------------------------------------*/
/*----< ncmpix_getn_int64_int() >--------------------------------------------*/
/*----< ncmpix_getn_int64_long() >-------------------------------------------*/
/*----< ncmpix_getn_int64_float() >------------------------------------------*/
/*----< ncmpix_getn_int64_double() >-----------------------------------------*/
/*----< ncmpix_getn_int64_ushort() >-----------------------------------------*/
/*----< ncmpix_getn_int64_uint() >-------------------------------------------*/
/*----< ncmpix_getn_int64_uint64() >-----------------------------------------*/
GETN_INT64(schar)
GETN_INT64(uchar)
GETN_INT64(short)
GETN_INT64(int)
GETN_INT64(long)
GETN_INT64(float)
GETN_INT64(double)
GETN_INT64(ushort)
GETN_INT64(uint)
GETN_INT64(uint64)

/*----< ncmpix_getn_int64_int64() >------------------------------------------*/
/* optimized version */
int
ncmpix_getn_int64_int64(const void **xpp, MPI_Offset nelems, int64 *tp)
{
# ifdef WORDS_BIGENDIAN
    memcpy(tp, *xpp, nelems * sizeof(int64));
# else
    ncmpii_swapn8b(tp, *xpp, nelems);
# endif
    *xpp = (const void *)((const char *)(*xpp) + nelems * X_SIZEOF_INT64);
    return NC_NOERR;
}

#define PUTN_INT64(btype)                                                     \
int                                                                           \
ncmpix_putn_int64_##btype(void **xpp, MPI_Offset nelems, const btype *tp)     \
{                                                                             \
    char *xp = (char *) *xpp;                                                 \
    int status = NC_NOERR;                                                    \
                                                                              \
    for ( ; nelems != 0; nelems--, xp += X_SIZEOF_INT64, tp++) {              \
        int lstatus = ncmpix_put_int64_##btype(xp, tp);                       \
        if (lstatus != NC_NOERR) status = lstatus;                            \
    }                                                                         \
                                                                              \
    *xpp = (void *)xp;                                                        \
    return status;                                                            \
}

/*----< ncmpix_putn_int64_schar() >------------------------------------------*/
/*----< ncmpix_putn_int64_uchar() >------------------------------------------*/
/*----< ncmpix_putn_int64_short() >------------------------------------------*/
/*----< ncmpix_putn_int64_int() >--------------------------------------------*/
/*----< ncmpix_putn_int64_long() >-------------------------------------------*/
/*----< ncmpix_putn_int64_float() >------------------------------------------*/
/*----< ncmpix_putn_int64_double() >-----------------------------------------*/
/*----< ncmpix_putn_int64_ushort() >-----------------------------------------*/
/*----< ncmpix_putn_int64_uint() >-------------------------------------------*/
/*----< ncmpix_putn_int64_uint64() >-----------------------------------------*/
PUTN_INT64(schar)
PUTN_INT64(uchar)
PUTN_INT64(short)
PUTN_INT64(int)
PUTN_INT64(long)
PUTN_INT64(float)
PUTN_INT64(double)
PUTN_INT64(ushort)
PUTN_INT64(uint)
PUTN_INT64(uint64)

/*----< ncmpix_putn_int64_int64() >------------------------------------------*/
/* optimized version */
int
ncmpix_putn_int64_int64(void **xpp, MPI_Offset nelems, const int64 *tp)
{
# ifdef WORDS_BIGENDIAN
    memcpy(*xpp, tp, nelems * X_SIZEOF_INT64);
# else
    ncmpii_swapn8b(*xpp, tp, nelems);
# endif
    *xpp = (void *)((char *)(*xpp) + nelems * X_SIZEOF_INT64);
    return NC_NOERR;
}


