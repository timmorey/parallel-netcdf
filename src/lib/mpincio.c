/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */

#include "ncconfig.h"

#include <unistd.h>  /* access() */
#include <assert.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <fcntl.h>
#include <string.h>
#ifdef _MSC_VER /* Microsoft Compilers */
#include <io.h>
#else
#include <unistd.h>
#endif

#include "nc.h"
#include "ncio.h"
#include "fbits.h"
#include "rnd.h"
#include "macro.h"

/* #define INSTRUMENT 1 */
#ifdef INSTRUMENT /* debugging */
#undef NDEBUG
#include <stdio.h>
#include "instr.h"
#endif

#if !defined(NDEBUG) && !defined(X_INT_MAX)
#define  X_INT_MAX INT_MAX
#endif

#if 0 /* !defined(NDEBUG) && !defined(X_ALIGN) */
#define  X_ALIGN 4
#else
#undef X_ALIGN
#endif

#define MAX_NC_ID 1024

static unsigned char IDalloc[MAX_NC_ID];

void
ncmpiio_free(ncio *nciop) {
    if (nciop != NULL) {
#ifdef HAVE_MPI_INFO_FREE
        if (nciop->mpiinfo != MPI_INFO_NULL)
            MPI_Info_free(&(nciop->mpiinfo));
#endif
        if (nciop->comm != MPI_COMM_NULL)
            MPI_Comm_free(&(nciop->comm));

        NCI_Free(nciop);
    }
}

ncio *
ncmpiio_new(const char *path, int ioflags)
{
  size_t sz_ncio = M_RNDUP(sizeof(ncio));
  size_t sz_path = M_RNDUP(strlen(path) +1); 
  ncio *nciop; 

  nciop = (ncio *) NCI_Malloc(sz_ncio + sz_path);
  if (nciop == NULL) 
    return NULL;

  nciop->ioflags = ioflags;
  nciop->mpiinfo = MPI_INFO_NULL;

  nciop->path = (char *) ((char *)nciop + sz_ncio);
  (void) strcpy((char *)nciop->path, path); 

  return nciop;
}

/*----< ncmpiio_extract_hints() >--------------------------------------------*/
/* this is where the I/O hints designated to pnetcdf are extracted */
static
void ncmpiio_extract_hints(ncio     *nciop,
                           MPI_Info  info)
{ 
    nciop->hints.header_align_size = 0;
    nciop->hints.var_align_size    = 0;

    /* extract NC hints */
    if (info != MPI_INFO_NULL) {
        char value[MPI_MAX_INFO_VAL];
        int  flag;

        MPI_Info_get(info, "nc_header_align_size", MPI_MAX_INFO_VAL-1, value, &flag);
        if (flag) nciop->hints.header_align_size = atoi(value);

        MPI_Info_get(info, "nc_var_align_size",    MPI_MAX_INFO_VAL-1, value, &flag);
        if (flag) nciop->hints.var_align_size = atoi(value);

        /* nc_header_align_size and nc_var_align_size take effect when a file
           is created or opened and later adding more header or variable data */

        if (nciop->hints.header_align_size < 0)
            nciop->hints.header_align_size = 0;
        if (nciop->hints.var_align_size < 0)
            nciop->hints.var_align_size = 0;
    }
}

/*----< ncmpiio_create() >---------------------------------------------------*/
int
ncmpiio_create(MPI_Comm     comm,
               const char  *path,
               int          ioflags,
               MPI_Info     info, 
               ncio       **nciopp)
{
    ncio *nciop;
    int i, rank, mpireturn; 
    int mpiomode = (MPI_MODE_RDWR | MPI_MODE_CREATE);
#ifdef NO_ACCESS
    int do_zero_file_size = 0;
#else
    int file_exist;
#endif

    /* TODO: use MPI_Allreduce to check for valid path, so error can be
       returned collectively */
    if (path == NULL || *path == 0)
        return NC_EINVAL;

    MPI_Comm_rank(comm, &rank);

    /* NC_CLOBBER is the default mode, even if it is not used in cmode */
#ifdef NO_ACCESS
    if (fIsSet(ioflags, NC_NOCLOBBER))
        fSet(mpiomode, MPI_MODE_EXCL);
    else
        do_zero_file_size = 1;
#else
    /* if access() is available, use it to check if file already exists */
    file_exist = 0;
    if (rank == 0) { /* root checks if file exists */
        /* remove the file system type prefix name if there is any.
         * For example, path=="lustre:/home/foo/testfile.nc",
         * use "/home/foo/testfile.nc" when calling access()
         */
        char *filename = strchr(path, ':');
        if (filename == NULL) /* no prefix */
            filename = (char*)path;
        else
            filename++;

        if (access(filename, F_OK) == 0) file_exist = 1;
    }

    if (fIsSet(ioflags, NC_NOCLOBBER)) {
        /* NetCDF requires NC_EEXIST returned when if the file already exists
         * and NC_NOCLOBBER is used in ioflags(cmode)
         */
        MPI_Bcast(&file_exist, 1, MPI_INT, 0, comm);
        if (file_exist) return NC_EEXIST;
    }
    else {
        /* to avoid calling MPI_File_set_size() later, let process 0 check
           if the file exists. If not, no need to call MPI_File_set_size */
        if (rank == 0 && file_exist) {
            /* file does exist, so delete it */
            mpireturn = MPI_File_delete((char*)path, MPI_INFO_NULL);
            if (mpireturn != MPI_SUCCESS) {
                ncmpii_handle_error(rank, mpireturn, "MPI_File_delete");
                return NC_EOFILE;
            }
        } /* else: the file does not exist, do nothing */
    }
#endif

    /* ignore if NC_NOWRITE set by user */
    fSet(ioflags, NC_WRITE);

    nciop = ncmpiio_new(path, ioflags); /* allocate buffer */
    if (nciop == NULL)
        return NC_ENOMEM;

    nciop->mpiomode  = MPI_MODE_RDWR;
    nciop->mpioflags = 0;
    MPI_Comm_dup(comm, &(nciop->comm));

    ncmpiio_extract_hints(nciop, info);

    mpireturn = MPI_File_open(nciop->comm, (char *)path, mpiomode, 
                              info, &nciop->collective_fh);
    if (mpireturn != MPI_SUCCESS) {
        /* NetCDF requires NC_EEXIST returned when if the file already exists
         * and NC_NOCLOBBER is used in ioflags(cmode)
         * In MPI 2.1, if MPI_File_open uses MPI_MODE_EXCL and the file
         * already exists, the error class MPI_ERR_FILE_EXISTS should
         * return. But in MPI 2.1 and prior, MPI_ERR_IO is returned.
         */
        int errorclass;
        ncmpiio_free(nciop);
#ifdef HAVE_MPI_ERR_FILE_EXISTS
        MPI_Error_class(mpireturn, &errorclass);
        if (errorclass == MPI_ERR_FILE_EXISTS) return NC_EEXIST;
#endif
        ncmpii_handle_error(rank, mpireturn, "MPI_File_open");
        return NC_EOFILE;
    }

    /* get the file info used by MPI-IO */
    MPI_File_get_info(nciop->collective_fh, &nciop->mpiinfo);

#ifdef NO_ACCESS
    if (do_zero_file_size) MPI_File_set_size(nciop->collective_fh, 0);
#endif

    for (i=0; i<MAX_NC_ID; i++)
        if (IDalloc[i] == 0)
            break;

    if (i == MAX_NC_ID) {
        ncmpiio_free(nciop);
        return NC_ENFILE;
    }

    *((int *)&nciop->fd) = i;
    IDalloc[i] = 1;

    /* collective I/O mode is the default mode */
    set_NC_collectiveFh(nciop);

    *nciopp = nciop;
    return NC_NOERR;  
}

/*----< ncmpiio_open() >-----------------------------------------------------*/
int
ncmpiio_open(MPI_Comm     comm,
             const char  *path,
             int          ioflags,
             MPI_Info     info,
             ncio       **nciopp)
{
    ncio *nciop;
    int i, mpireturn;
    int mpiomode = fIsSet(ioflags, NC_WRITE) ? MPI_MODE_RDWR : MPI_MODE_RDONLY;

    /* TODO: use MPI_Allreduce to check for valid path, so error can be
       returned collectively */
    if (path == NULL || *path == 0)
        return NC_EINVAL;
 
    nciop = ncmpiio_new(path, ioflags);
    if (nciop == NULL)
        return NC_ENOMEM;
 
    nciop->mpiomode  = mpiomode;
    nciop->mpioflags = 0;
    MPI_Comm_dup(comm, &(nciop->comm));
 
    ncmpiio_extract_hints(nciop, info);

    mpireturn = MPI_File_open(nciop->comm, (char *)path, mpiomode,
                              info, &nciop->collective_fh);
    if (mpireturn != MPI_SUCCESS) {
        int rank, errorclass;
        ncmpiio_free(nciop);
#ifdef HAVE_MPI_ERR_NO_SUCH_FILE
        MPI_Error_class(mpireturn, &errorclass);
        if (errorclass == MPI_ERR_NO_SUCH_FILE) return NC_ENOENT;
#endif
        MPI_Comm_rank(comm, &rank);
        ncmpii_handle_error(rank, mpireturn, "MPI_File_open");
        return NC_EOFILE;
    }

    /* get the file info used by MPI-IO */
    MPI_File_get_info(nciop->collective_fh, &nciop->mpiinfo);
 
    for (i = 0; i < MAX_NC_ID && IDalloc[i] != 0; i++);
    if (i == MAX_NC_ID) {
        ncmpiio_free(nciop);
        return NC_ENFILE;
    }
    *((int *)&nciop->fd) = i;
    IDalloc[i] = 1;
 
    set_NC_collectiveFh(nciop);
 
    *nciopp = nciop;
    return NC_NOERR; 
}

int
ncmpiio_sync(ncio *nciop) {
    int mpireturn;

    if (NC_independentFhOpened(nciop)) {
        mpireturn = MPI_File_sync(nciop->independent_fh);
        if (mpireturn != MPI_SUCCESS) {
            int rank;
            MPI_Comm_rank(nciop->comm, &rank);
            ncmpii_handle_error(rank, mpireturn, "MPI_File_sync");
            return NC_EFILE;
        }
    }
    if (NC_collectiveFhOpened(nciop)) {
        mpireturn = MPI_File_sync(nciop->collective_fh);
        if (mpireturn != MPI_SUCCESS) {
            int rank;
            MPI_Comm_rank(nciop->comm, &rank);
            ncmpii_handle_error(rank, mpireturn, "MPI_File_sync");
            return NC_EFILE;
        }
    }
    MPI_Barrier(nciop->comm);

    return NC_NOERR;
}

int
ncmpiio_close(ncio *nciop, int doUnlink) {
  int status = NC_NOERR;
  int mpireturn;

  if (nciop == NULL) /* this should never occur */
    return NC_EINVAL;

  if(NC_independentFhOpened(nciop)) {
    mpireturn = MPI_File_close(&(nciop->independent_fh));
    if (mpireturn != MPI_SUCCESS) {
      int rank;
      MPI_Comm_rank(nciop->comm, &rank);
      ncmpii_handle_error(rank, mpireturn, "MPI_File_close");
      return NC_EFILE;
    }
  }

 
  if(NC_collectiveFhOpened(nciop)) {
    mpireturn = MPI_File_close(&(nciop->collective_fh));  
    if (mpireturn != MPI_SUCCESS) {
      int rank;
      MPI_Comm_rank(nciop->comm, &rank);
      ncmpii_handle_error(rank, mpireturn, "MPI_File_close");
      return NC_EFILE;
    }
  }
  IDalloc[*((int *)&nciop->fd)] = 0;

  if (doUnlink) {
    mpireturn = MPI_File_delete((char *)nciop->path, nciop->mpiinfo);
/*
    if (mpireturn != MPI_SUCCESS) {
      char errorString[512];
      int  errorStringLen;
      int rank;
      MPI_Comm_rank(nciop->comm, &rank);
      MPI_Error_string(mpireturn, errorString, &errorStringLen);
      printf("%2d: MPI_File_delete error = %s\n", rank, errorString);
      return NC_EFILE;
    }
*/
  }
  ncmpiio_free(nciop);

  return status;
}

/*----< ncmpiio_move() >-----------------------------------------------------*/
int
ncmpiio_move(ncio *const nciop,
             MPI_Offset  to,
             MPI_Offset  from,
             MPI_Offset  nbytes)
{
    int rank, grpsize, mpireturn;
    void *buf;
    const MPI_Offset bufsize = 4096;
    MPI_Offset movesize, bufcount;
    MPI_Status mpistatus;

    MPI_Comm_size(nciop->comm, &grpsize);
    MPI_Comm_rank(nciop->comm, &rank);

    movesize = nbytes;
    buf = NCI_Malloc((size_t)bufsize);
    if (buf == NULL)
        return NC_ENOMEM;

    while (movesize > 0) {
        /* find a proper number of processors to participate I/O */
        while (grpsize > 1 && movesize/grpsize < bufsize)
            grpsize--;
        if (grpsize > 1) {
            bufcount = bufsize;
            movesize -= bufsize*grpsize;
        } 
        else if (movesize < bufsize) {
            bufcount = movesize;
            movesize = 0;
        } 
        else {
            bufcount = bufsize;
            movesize -= bufsize;
        }

        /* fileview is always entire file visible */

        if (rank >= grpsize) bufcount = 0;
        /* read the original data @ from+movesize+rank*bufsize */
        mpireturn = MPI_File_read_at_all(nciop->collective_fh,
                                         from+movesize+rank*bufsize,
                                         buf, bufcount, MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
	    ncmpii_handle_error(rank, mpireturn, "MPI_File_read_at");
            NCI_Free(buf);
            return NC_EREAD;
        }

        MPI_Barrier(nciop->comm); /* important, in case new region overlaps old region */

        if (rank >= grpsize) bufcount = 0;
        /* write to new location @ to+movesize+rank*bufsize */
        mpireturn = MPI_File_write_at_all(nciop->collective_fh,
                                          to+movesize+rank*bufsize,
                                          buf, bufcount, MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
	    ncmpii_handle_error(rank, mpireturn, "MPI_File_write_at");
            NCI_Free(buf);
            return NC_EWRITE;
        }
    }
    NCI_Free(buf);
    return NC_NOERR;
}

int ncmpiio_get_hint(NC *ncp, char *key, char *value, int *flag)
{
    MPI_Info info;

    /* info hints can come from the file system but can also come from
     * user-specified hints.  the MPI implementation probably should
     * merge the two, but some implementaitons not only ignore hints
     * they don't understand, but also fail to incorporate those hints
     * into the info struct (this is unfortunate for us, but entirely
     * standards compilant). 
     *
     * Our policy will be to use the implementation's info first
     * (perhaps the implementaiton knows something about the underlying
     * file system), and then consult user-supplied hints should we not
     * find the hint in the info associated with the MPI file descriptor
     */

    /* first check the hint from the MPI library ... */
    MPI_File_get_info(ncp->nciop->collective_fh, &info);
    if (info != MPI_INFO_NULL) 
        MPI_Info_get(info, key, MPI_MAX_INFO_VAL-1, value, flag);
    if (*flag == 0)  {
        /* ... then check the hint passed in through ncmpi_create */
        if (ncp->nciop->mpiinfo != MPI_INFO_NULL) {
            MPI_Info_get(ncp->nciop->mpiinfo, key, 
                    MPI_MAX_INFO_VAL-1, value, flag);
        }
    }
    if (info != MPI_INFO_NULL) 
        MPI_Info_free(&info);

    return 0;
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 * End:
 *
 * vim: ts=8 sts=4 sw=4 expandtab
 */
