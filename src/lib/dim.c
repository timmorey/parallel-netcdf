/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: dim.c 989 2012-03-11 03:00:30Z wkliao $ */

#include "nc.h"
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <string.h>
#include <assert.h>
#include "ncx.h"
#include "fbits.h"
#include "macro.h"

/*
 * Free dim
 * Formerly
NC_free_dim(dim)
 */
void
ncmpii_free_NC_dim(NC_dim *dimp)
{
	if(dimp == NULL)
		return;
	ncmpii_free_NC_string(dimp->name);
	NCI_Free(dimp);
}


/* allocate and return a new NC_dim object */
NC_dim *
ncmpii_new_x_NC_dim(NC_string *name)
{
    NC_dim *dimp;

    dimp = (NC_dim *) NCI_Malloc(sizeof(NC_dim));
    if (dimp == NULL) return NULL;

    dimp->name = name;
    dimp->size = 0;

    return(dimp);
}

/*----< ncmpii_new_NC_dim() >-----------------------------------------------*/
/*
 * Formerly, NC_new_dim(const char *name, long size)
 */
static NC_dim *
ncmpii_new_NC_dim(const char *name,
                  MPI_Offset  size)
{
    NC_string *strp;
    NC_dim *dimp;

    strp = ncmpii_new_NC_string(strlen(name), name);
    if (strp == NULL) return NULL;

    dimp = ncmpii_new_x_NC_dim(strp);
    if (dimp == NULL) {
    	ncmpii_free_NC_string(strp);
    	return NULL;
    }

    dimp->size = size;

    return(dimp);
}


static NC_dim *
dup_NC_dim(const NC_dim *dimp)
{
	return ncmpii_new_NC_dim(dimp->name->cp, dimp->size);
}

/*----< ncmpii_find_NC_Udim() >----------------------------------------------*/
/*
 * Step thru NC_DIMENSION array, seeking the UNLIMITED dimension.
 * Return dimid or -1 on not found.
 * *dimpp is set to the appropriate NC_dim.
 */
int
ncmpii_find_NC_Udim(const NC_dimarray  *ncap,
                    NC_dim            **dimpp)
{
    int dimid;

    assert(ncap != NULL);

    if (ncap->ndefined == 0) return -1;

    /* note that the number of dimensions allowed is < 2^32 */
    for (dimid=0; dimid<ncap->ndefined; dimid++)
        if (ncap->value[dimid]->size == NC_UNLIMITED) {
            /* found the mateched name */
            if (dimpp != NULL)
                *dimpp = ncap->value[dimid];
            return dimid;
        }

    /* not found */
    return -1;
}

/*----< NC_finddim() >-------------------------------------------------------*/
/*
 * Step thru NC_DIMENSION array, seeking match on name.
 * Return dimid or -1 on not found.
 * *dimpp is set to the appropriate NC_dim.
 */
static int
NC_finddim(const NC_dimarray  *ncap,
           const char         *name,
           NC_dim            **dimpp)
{
    int dimid;

    assert(ncap != NULL);

    if (ncap->ndefined == 0) return -1;

    /* note that the number of dimensions allowed is < 2^32 */
    for (dimid=0; dimid<ncap->ndefined; dimid++)
        if (strcmp(ncap->value[dimid]->name->cp, name) == 0) {
            /* found the mateched name */
            if (dimpp != NULL)
                *dimpp = ncap->value[dimid];
            return dimid;
        }

    /* the name is not found */
    return -1;
}


/* dimarray */


/*
 * Free the stuff "in" (referred to by) an NC_dimarray.
 * Leaves the array itself allocated.
 */
void
ncmpii_free_NC_dimarrayV0(NC_dimarray *ncap)
{
	assert(ncap != NULL);

	if(ncap->ndefined == 0)
		return;

	assert(ncap->value != NULL);

	{
		NC_dim **dpp = ncap->value;
		NC_dim *const *const end = &dpp[ncap->ndefined];
		for( /*NADA*/; dpp < end; dpp++)
		{
			ncmpii_free_NC_dim(*dpp);
			*dpp = NULL;
		}
	}
	ncap->ndefined = 0;
}


/*
 * Free NC_dimarray values.
 * formerly
NC_free_array()
 */
void
ncmpii_free_NC_dimarrayV(NC_dimarray *ncap)
{
	assert(ncap != NULL);
	
	if(ncap->nalloc == 0)
		return;

	assert(ncap->value != NULL);

	ncmpii_free_NC_dimarrayV0(ncap);

	NCI_Free(ncap->value);
	ncap->value = NULL;
	ncap->nalloc = 0;
}


int
ncmpii_dup_NC_dimarrayV(NC_dimarray *ncap, const NC_dimarray *ref)
{
	int status = NC_NOERR;

	assert(ref != NULL);
	assert(ncap != NULL);

	if(ref->ndefined != 0)
	{
		const MPI_Offset sz = ref->ndefined * sizeof(NC_dim *);
		ncap->value = (NC_dim **) NCI_Malloc(sz);
		if(ncap->value == NULL)
			return NC_ENOMEM;
		(void) memset(ncap->value, 0, sz);
		ncap->nalloc = ref->ndefined;
	}

	ncap->ndefined = 0;
	{
		NC_dim **dpp = ncap->value;
		const NC_dim **drpp = (const NC_dim **)ref->value;
		NC_dim *const *const end = &dpp[ref->ndefined];
		for( /*NADA*/; dpp < end; drpp++, dpp++, ncap->ndefined++)
		{
			*dpp = dup_NC_dim(*drpp);
			if(*dpp == NULL)
			{
				status = NC_ENOMEM;
				break;
			}
		}
	}

	if(status != NC_NOERR)
	{
		ncmpii_free_NC_dimarrayV(ncap);
		return status;
	}

	assert(ncap->ndefined == ref->ndefined);

	return NC_NOERR;
}


/*----< incr_NC_dimarray() >------------------------------------------------*/
/*
 * Add a new handle to the end of an array of handles
 * Formerly, NC_incr_array(array, tail)
 */
static int
incr_NC_dimarray(NC_dimarray *ncap,
                 NC_dim      *newdimp)
{
    NC_dim **vp;

    assert(ncap != NULL);

    if (ncap->nalloc == 0) {
        assert(ncap->ndefined == 0);
        vp = (NC_dim **) NCI_Malloc(NC_ARRAY_GROWBY * sizeof(NC_dim *));
        if (vp == NULL) return NC_ENOMEM;

        ncap->value = vp;
        ncap->nalloc = NC_ARRAY_GROWBY;
    }
    else if (ncap->ndefined + 1 > ncap->nalloc) {
        vp = (NC_dim **) NCI_Realloc(ncap->value,
             (ncap->nalloc + NC_ARRAY_GROWBY) * sizeof(NC_dim *));
        if (vp == NULL) return NC_ENOMEM;

        ncap->value = vp;
        ncap->nalloc += NC_ARRAY_GROWBY;
    }
    /* else here means some space still available */

    if (newdimp != NULL) {
        ncap->value[ncap->ndefined] = newdimp;
        ncap->ndefined++;
    }

    return NC_NOERR;
}


/*----< ncmpii_elem_NC_dimarray() >------------------------------------------*/
NC_dim *
ncmpii_elem_NC_dimarray(const NC_dimarray *ncap,
                        size_t             dimid)
{
    /* returns the dimension ID defined earlier */
    assert(ncap != NULL);

    if (ncap->ndefined == 0 || dimid >= ncap->ndefined)
        return NULL;

    assert(ncap->value != NULL);

    return ncap->value[dimid];
}


/* Public */

/*----< ncmpi_def_dim() >---------------------------------------------------*/
int
ncmpi_def_dim(int         ncid,    /* IN:  file ID */
              const char *name,    /* IN:  name of dimension */
              MPI_Offset  size,    /* IN:  dimension size */
              int        *dimidp)  /* OUT: dimension ID */
{
    int dimid, file_ver, status;
    NC *ncp;
    NC_dim *dimp;

    /* check if ncid is valid */
    status = ncmpii_NC_check_id(ncid, &ncp); 
    if (status != NC_NOERR) return status;

    /* check if called in define mode */
    if (!NC_indef(ncp)) return NC_ENOTINDEFINE;

    /* check if the name string is legal for netcdf format */
    file_ver = 1;
    if (fIsSet(ncp->flags, NC_64BIT_OFFSET))
        file_ver = 2;
    else if (fIsSet(ncp->flags, NC_64BIT_DATA))
        file_ver = 5;

    status = ncmpii_NC_check_name(name, file_ver);
    if (status != NC_NOERR) return status;

    /* MPI_Offset is usually a signed value, but serial netcdf uses 
     * MPI_Offset -- normally unsigned */
    if ((ncp->flags & NC_64BIT_OFFSET) && sizeof(off_t) > 4) {
        /* CDF2 format and LFS */
        if (size > X_UINT_MAX - 3 || (size < 0)) 
            /* "-3" handles rounded-up size */
            return NC_EDIMSIZE;
    } else if ((ncp->flags & NC_64BIT_DATA)) {
        /* CDF5 format*/
        if (size < 0) 
            return NC_EDIMSIZE;
    } else {
        /* CDF1 format */
        if (size > X_INT_MAX - 3 || (size < 0))
            /* "-3" handles rounded-up size */
            return NC_EDIMSIZE;
    }

    if (size == NC_UNLIMITED) {
        /* check for any existing unlimited dimension, netcdf allows
         * one per file
         */
        dimid = ncmpii_find_NC_Udim(&ncp->dims, &dimp);
        if (dimid != -1) return NC_EUNLIMIT; /* found an existing one */
    }

    /* check if exceeds the upperbound has reached */
    if (ncp->dims.ndefined >= NC_MAX_DIMS) return NC_EMAXDIMS;

    /* check if the name string is previously used */
    dimid = NC_finddim(&ncp->dims, name, &dimp);
    if (dimid != -1) return NC_ENAMEINUSE;
    
    /* create a new dimension object */
    dimp = ncmpii_new_NC_dim(name, size);
    if (dimp == NULL) return NC_ENOMEM;

    /* Add a new handle to the end of an array of handles */
    status = incr_NC_dimarray(&ncp->dims, dimp);
    if (status != NC_NOERR) {
        ncmpii_free_NC_dim(dimp);
        return status;
    }

    if (dimidp != NULL)
        *dimidp = (int)ncp->dims.ndefined -1;
        /* ncp->dims.ndefined has been increased in incr_NC_dimarray() */

    return NC_NOERR;
}


int
ncmpi_inq_dimid(int ncid, const char *name, int *dimid_ptr)
{
	int status;
	NC *ncp;
	int dimid;

	status = ncmpii_NC_check_id(ncid, &ncp); 
	if(status != NC_NOERR)
		return status;

	dimid = NC_finddim(&ncp->dims, name, NULL);

	if(dimid == -1)
		return NC_EBADDIM;

	*dimid_ptr = dimid;
	return NC_NOERR;
}


int
ncmpi_inq_dim(int ncid, int dimid, char *name, MPI_Offset *sizep)
{
	int status;
	NC *ncp;
	NC_dim *dimp;

	status = ncmpii_NC_check_id(ncid, &ncp); 
	if(status != NC_NOERR)
		return status;

	dimp = ncmpii_elem_NC_dimarray(&ncp->dims, (size_t) dimid);
	if(dimp == NULL)
		return NC_EBADDIM;

	if(name != NULL)
	{
		(void)strncpy(name, dimp->name->cp, 
			dimp->name->nchars);
		name[dimp->name->nchars] = 0;
	}
	if(sizep != 0)
	{
		if(dimp->size == NC_UNLIMITED)
			*sizep = NC_get_numrecs(ncp);
		else
			*sizep = dimp->size;	
	}
	return NC_NOERR;
}


int 
ncmpi_inq_dimname(int ncid, int dimid, char *name)
{
	int status;
	NC *ncp;
	NC_dim *dimp;

	status = ncmpii_NC_check_id(ncid, &ncp); 
	if(status != NC_NOERR)
		return status;

	dimp = ncmpii_elem_NC_dimarray(&ncp->dims, (size_t) dimid);
	if(dimp == NULL)
		return NC_EBADDIM;

	if(name != NULL)
	{
		(void)strncpy(name, dimp->name->cp, 
			dimp->name->nchars);
		name[dimp->name->nchars] = 0;
	}

	return NC_NOERR;
}


int 
ncmpi_inq_dimlen(int ncid, int dimid, MPI_Offset *lenp)
{
	int status;
	NC *ncp;
	NC_dim *dimp;

	status = ncmpii_NC_check_id(ncid, &ncp); 
	if(status != NC_NOERR)
		return status;

	dimp = ncmpii_elem_NC_dimarray(&ncp->dims, (size_t) dimid);
	if(dimp == NULL)
		return NC_EBADDIM;

	if(lenp != 0)
	{
		if(dimp->size == NC_UNLIMITED)
			*lenp = NC_get_numrecs(ncp);
		else
			*lenp = dimp->size;	
	}
	return NC_NOERR;
}


int
ncmpi_rename_dim( int ncid, int dimid, const char *newname)
{
    int file_ver, status, existid;
    NC *ncp;
    NC_dim *dimp;

    status = ncmpii_NC_check_id(ncid, &ncp); 
    if (status != NC_NOERR)
        return status;

    if (NC_readonly(ncp))
        return NC_EPERM;

    file_ver = 1;
    if (fIsSet(ncp->flags, NC_64BIT_OFFSET))
        file_ver = 2;
    else if (fIsSet(ncp->flags, NC_64BIT_DATA))
        file_ver = 5;

    status = ncmpii_NC_check_name(newname, file_ver);
    if (status != NC_NOERR) return status;

    existid = NC_finddim(&ncp->dims, newname, &dimp);
    if (existid != -1)
        return NC_ENAMEINUSE;

    dimp = ncmpii_elem_NC_dimarray(&ncp->dims, (size_t) dimid);
    if (dimp == NULL)
        return NC_EBADDIM;

    if (NC_indef(ncp)) {
        NC_string *old = dimp->name;
        NC_string *newStr = ncmpii_new_NC_string(strlen(newname), newname);
        if (newStr == NULL)
            return NC_ENOMEM;
        dimp->name = newStr;
        ncmpii_free_NC_string(old);
        return NC_NOERR;
    }
    /* else, not in define mode */

    status = ncmpii_set_NC_string(dimp->name, newname);
    if (status != NC_NOERR)
        return status;

    set_NC_hdirty(ncp);

    if (NC_doHsync(ncp)) {
        status = ncmpii_NC_sync(ncp, 1);
        if (status != NC_NOERR)
            return status;
    }

    return NC_NOERR;
}
