/**
   pism.c - Created by Timothy Morey on 11/26/2012

   This file contains code to help along with the PISM optimizations in the 
   parallel-netcdf library.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pism.h"
#include "macro.h"

//#define WRITE_DEBUG_MESSAGES 1
#define MEASURE_PERF 1
#define MAX_VARPIECES 512
#define MAX_FILEPIECES 512
#define MAX_LUSTREPIECES 1024
#define MAX_REQS 32768

int DoLustreOptimizedWrite(NC* ncp, NC_var* varp, 
                           const MPI_Offset start[], 
                           const MPI_Offset count[],
                           void* xbuf, MPI_File fh) {
  int retval = NC_NOERR;

  MPI_Offset localmetadata[16];
  MPI_Offset* metadatabuf = 0;
  MPI_Offset metadatalen = 0;

  MPI_Offset varoffset[MAX_VARPIECES], varlength[MAX_VARPIECES];
  int nvarpieces = MAX_VARPIECES;
  
  MPI_Offset fileoffset[MAX_FILEPIECES], filelength[MAX_FILEPIECES];
  int nfilepieces = MAX_FILEPIECES;
  
  MPI_Offset **lustreoffset, **lustrelength;
  int *nlustrepieces;
  
  int rank, commsize;
  int i;
        
  MPI_Datatype view;
  MPI_Offset xbuf_offset;

  MPI_Info info;
  char hintvalue[MPI_MAX_INFO_VAL];
  int hintfound;
  
  MPI_Offset stripesize = 1048576;
  MPI_Offset stripecount = 4;

  int coratio = 1;
  int* writers;
  char** stripes;
  int nstripes;

#ifdef MEASURE_PERF
  double starttime, initend, mapend, redistend, writeend;
#endif

#ifdef MEASURE_PERF
  starttime = MPI_Wtime();
#endif

  MPI_Comm_rank(ncp->nciop->comm, &rank);
  MPI_Comm_size(ncp->nciop->comm, &commsize);

  MPI_File_get_info(fh, &info);
  MPI_Info_get(info, "striping_unit", 
               MPI_MAX_INFO_VAL, hintvalue, &hintfound);
  if(hintfound) {
    stripesize = atoi(hintvalue);
  } else {
    fprintf(stderr, "Rank %03d: Unable to find the Lustre stripe size,"
            " defaulting to %d.\n",
            rank, (int)stripesize);
  }

  MPI_Info_get(info, "striping_factor", 
               MPI_MAX_INFO_VAL, hintvalue, &hintfound);
  if(hintfound) {
    stripecount = atoi(hintvalue);
  } else {
    fprintf(stderr, "Rank %03d: Unable to find the Lustre stripe count,"
            " defaulting to %d.\n",
            rank, (int)stripecount);
  }

  if(ncp->nciop->hints.pism_co_ratio == -1) {
    // This hint usually isn't set and should be considered optional, so don't
    // complain if it isn't set.
    //fprintf(stderr, "Rank %03d: Unable to find CO ratio, "
    //        " defaulting to %d.\n",
    //        rank, coratio);
  } else {
    coratio = ncp->nciop->hints.pism_co_ratio;
  }

#ifdef WRITE_DEBUG_MESSAGES
  printf("Rank %03d: Collective write for '%s'\n", rank, varp->name->cp);
  printf("Rank %03d: xsz=%d, len=%d, begin=%d\n", 
         rank, (int)varp->xsz, (int)varp->len, (int)varp->begin);
  printf("Rank %04d: stripesize=%d, stripecount=%d, coratio=%d\n",
	 rank, (int)stripesize, (int)stripecount, coratio);

  if(1 == varp->ndims) {
    printf("Rank %03d: Dataspace start=(%d), count=(%d)\n", 
           rank, (int)start[0], (int)count[0]);
  } else if(2 == varp->ndims) {
    printf("Rank %03d: Dataspace start=(%d, %d), count=(%d, %d)\n", 
           rank, (int)start[0], (int)start[1], 
           (int)count[0], (int)count[1]);
  } else if(3 == varp->ndims) {
    printf("Rank %03d: Dataspace start=(%d, %d, %d), count=(%d, %d, %d)\n", 
           rank, (int)start[0], (int)start[1], (int)start[2], 
           (int)count[0], (int)count[1], (int)count[2]);
  } else if(4 == varp->ndims) {
    printf("Rank %03d: Dataspace start=(%d, %d, %d, %d), "
           "count=(%d, %d, %d, %d)\n", 
           rank, 
           (int)start[0], (int)start[1], (int)start[2], (int)start[3], 
           (int)count[0], (int)count[1], (int)count[2], (int)count[3]);
  } else {
    fprintf(stderr, "Rank%03d: Unexpected number of dimensions: %d!!!\n", 
            rank, varp->ndims);
  }
#endif

  writers = malloc(stripecount * coratio * sizeof(int));
  SelectWriters(ncp->nciop->comm, stripecount * coratio, writers);

  lustreoffset = malloc(commsize * sizeof(MPI_Offset*));
  lustrelength = malloc(commsize * sizeof(MPI_Offset*));
  nlustrepieces = malloc(commsize * sizeof(int));

  // TODO: we're creating far more stripe buffers than we probably need:
  nstripes = ((varp->begin + varp->len) / stripesize) + 1;
  stripes = malloc(nstripes * sizeof(char*));
  memset(stripes, 0, nstripes * sizeof(char*));

  for(i = 0; i < varp->ndims; i++) {
    localmetadata[i] = start[i];
    localmetadata[varp->ndims + i] = count[i];
  }

  metadatalen = 2 * varp->ndims * commsize;
  metadatabuf = malloc(metadatalen * sizeof(MPI_Offset));

#ifdef MEASURE_PERF
  initend = MPI_Wtime();
#endif

  MPI_Allgather(localmetadata, 2 * varp->ndims, MPI_OFFSET,
                metadatabuf, 2 * varp->ndims, MPI_OFFSET, 
                ncp->nciop->comm);

  for(i = 0; i < commsize; i++) {
    MPI_Offset *s, *c;

    s = metadatabuf + (i * (2 * varp->ndims));
    c = metadatabuf + (i * (2 * varp->ndims) + varp->ndims);

    nvarpieces = MAX_VARPIECES;
    nfilepieces = MAX_FILEPIECES;
    nlustrepieces[i] = MAX_LUSTREPIECES;
    lustreoffset[i] = malloc(MAX_LUSTREPIECES * sizeof(MPI_Offset));
    lustrelength[i] = malloc(MAX_LUSTREPIECES * sizeof(MPI_Offset));

    if(NC_NOERR != 
       (retval = DataSpaceToVarSpace(ncp, varp, s, c, 
                                     varoffset, varlength, &nvarpieces))) {
      fprintf(stderr, "Rank %03d: DataSpaceToVarSpace failed.\n", rank);
    }
    
    if(NC_NOERR != 
       (retval = VarSpaceToFileSpace(ncp, varp, 
                                     varoffset, varlength, nvarpieces,
                                     fileoffset, filelength, &nfilepieces))) {
      fprintf(stderr, "Rank %03d: VarSpaceToFileSpace failed.\n", rank);
    }
    
    if(NC_NOERR != 
       (retval = FileSpaceToLustreSpace(ncp, varp,
                                        stripesize, stripecount,
                                        fileoffset, filelength, nfilepieces,
                                        lustreoffset[i], lustrelength[i], 
                                        &nlustrepieces[i]))) {
      fprintf(stderr, "Rank %03d: FileSpaceToLustreSpace failed.\n", rank);
    }
  }

#ifdef MEASURE_PERF
  mapend = MPI_Wtime();
#endif

  if(NC_NOERR !=
     (retval = Redistribute(ncp, varp, 
                            stripesize, stripecount, coratio,
                            lustreoffset, lustrelength, nlustrepieces, xbuf, 
                            writers, stripes, nstripes))) {
    fprintf(stderr, "Rank %03d: Redistribute failed.\n", rank);
  }

#ifdef MEASURE_PERF
  redistend = MPI_Wtime();
#endif

  if(NC_NOERR !=
     (retval = Write(ncp, varp, fh, 
                     stripes, nstripes, stripesize, stripecount))) {
    fprintf(stderr, "Rank %03d: Write failed.\n", rank);
  }

#ifdef MEASURE_PERF
  writeend = MPI_Wtime();

  printf("Rank %03d: (%s) init-time   = %8.6f s\n", 
         rank, varp->name->cp, initend - starttime);
  printf("Rank %03d: (%s) map-time    = %8.6f s\n", 
         rank, varp->name->cp, mapend - initend);
  printf("Rank %03d: (%s) redist-time = %8.6f s\n", 
         rank, varp->name->cp, redistend - mapend);
  printf("Rank %03d: (%s) write-time  = %8.6f s\n", 
         rank, varp->name->cp, writeend - redistend);
#endif

  for(i = 0; i < nstripes; i++) {
    if(stripes[i])
      free(stripes[i]);
  }

  free(stripes);
  free(metadatabuf);

  for(i = 0; i < commsize; i++) {
    free(lustreoffset[i]);
    free(lustrelength[i]);
  }

  free(lustreoffset);
  free(lustrelength);
  free(nlustrepieces);

  free(writers);

  return retval;
}

int DataSpaceToVarSpace(NC* ncp, NC_var* varp, 
                        const MPI_Offset start[],
                        const MPI_Offset count[],
                        MPI_Offset varoffset[],
                        MPI_Offset varlength[],
                        int* len) {
  int retval = NC_NOERR;
  MPI_Offset varsize, offset, maxoffset;
  MPI_Offset local_stripesize, global_stripesize;
  int maxlen = *len;
  int i;
  int unlimdim;
  MPI_Offset size;

  unlimdim = ncmpii_find_NC_Udim(&ncp->dims, NULL);

  offset = 0;
  maxoffset = 0;
  for(i = varp->ndims - 1; i >= 0; i--) {

    if(i == varp->ndims - 1) {
      size = 1;
    } else {
      size = varp->dsizes[i+1];
    }

    // We don't want to consider the unlimdim here, because we are only 
    // considering writes for non-record variables and single record writes for
    // record variables.  For record variables, the offsets we are caclulating
    // here are internal to a record.
    if(varp->dimids[i] != unlimdim) {
      offset += start[i] * size;
      maxoffset += (start[i] + count[i] - 1) * size;
    }
  }
  
  local_stripesize = 1;
  global_stripesize = 1;
  for(i = varp->ndims - 1; i >= 0; i--) {
    // We need to avoid the empty unlimdim here, which has a 'shape' of 0.
    if(varp->dimids[i] != unlimdim) {
      global_stripesize *= varp->shape[i];
      
      if(count[i] >= varp->shape[i]) {
        local_stripesize *= varp->shape[i];
      } else {
        local_stripesize *= count[i];
        break;
      }
    }
  }

  *len = 0;
  while(offset < maxoffset) {
    varoffset[*len] = offset;
    varlength[*len] = local_stripesize;

    (*len)++;
    offset += global_stripesize;

    if(*len >= maxlen) {
      retval = -1;
      fprintf(stderr, "DataSpaceToVarSpace: too many pieces\n");
      break;
    }
  }

  return retval;
}

int VarSpaceToFileSpace(NC* ncp, NC_var* varp,
                        const MPI_Offset varoffset[], 
                        const MPI_Offset varlength[],
                        int nvarpieces,
                        MPI_Offset fileoffset[],
                        MPI_Offset filelength[],
                        int* nfilepieces) {
  int retval = 0;
  int i;

  /* TODO: For record variables, we always assume we are writing record 0 */

  if(*nfilepieces < nvarpieces) {
    retval = -1;
    fprintf(stderr, "VarSpaceToFileSpace: too many pieces\n");
  } else {
    *nfilepieces = nvarpieces;
  }

  for(i = 0; i < nvarpieces && 0 == retval; i++) {
    fileoffset[i] = varp->begin + varoffset[i] * varp->xsz;
    filelength[i] = varlength[i] * varp->xsz;
  }

  return retval;
}

int FileSpaceToLustreSpace(NC* ncp, NC_var* varp,
                           MPI_Offset stripesize, MPI_Offset stripecount,
                           const MPI_Offset fileoffset[],
                           const MPI_Offset filelength[],
                           int n_filepieces,
                           MPI_Offset lustreoffset[],
                           MPI_Offset lustrelength[],
                           int* nlustrepieces) {
  int retval = 0;
  int maxpieces = *nlustrepieces;
  MPI_Offset offset, length, stripeoffset;
  int i;

  *nlustrepieces = 0;
  for(i = 0; i < n_filepieces; i++) {
    offset = fileoffset[i];
    while(offset < fileoffset[i] + filelength[i]) {
      stripeoffset = offset % stripesize;
      length = MIN(stripesize - stripeoffset,
                   (fileoffset[i] + filelength[i]) - offset);
      
      lustreoffset[*nlustrepieces] = offset;
      lustrelength[*nlustrepieces] = length;
      
      offset += length;
      (*nlustrepieces)++;
      
      if(*nlustrepieces > maxpieces) {
        retval = -1;
        fprintf(stderr, "FileSpaceToLustreSpace: too many pieces\n");
        break;
      }
    }
  }

  return retval;
}

int SelectWriters(MPI_Comm comm, int stripecount, int writers[]) {
  int retval = 0;
  int size, stride;
  int i;

  MPI_Comm_size(comm, &size);
  
  stride = size / stripecount;

  writers[0] = 0;
  for(i = 1; i < stripecount; i++) {
    writers[i] = writers[i-1] + stride;
  }

  return retval;
}

int Redistribute(NC* ncp, NC_var* varp,
                 int stripesize, int stripecount, int coratio,
                 MPI_Offset** lustreoffset,
                 MPI_Offset** lustrelength,
                 const int npieces[],
                 const void* xbuf,
                 const int writers[],
                 char* stripes[],
                 int nstripes) {
  int retval = 0;
  int rank, size;
  int i, j, k;
  MPI_Offset offset, length;
  MPI_Offset stripe;
  MPI_Offset xbufoffset, stripeoffset;
  MPI_Offset varoffset, varlength;
  MPI_Offset localbytes, writebytes;
  MPI_Request asyncreqs[MAX_REQS];
  int reqcount = 0;
  int nsendbufs;
  char** sendbuf;
  MPI_Offset* sendbuflen;
  MPI_Offset sendbufoffset;
  int nrecvbufs;
  char** recvbuf;
  MPI_Offset* recvbuflen;
  MPI_Offset recvbufoffset;

  MPI_Comm_rank(ncp->nciop->comm, &rank);
  MPI_Comm_size(ncp->nciop->comm, &size);

  varoffset = varp->begin;
  varlength = varp->len;

  localbytes = 0;
  writebytes = 0;

  nsendbufs = stripecount * coratio;
  sendbuf = malloc(nsendbufs * sizeof(char*));
  memset(sendbuf, 0, nsendbufs * sizeof(char*));
  sendbuflen = malloc(nsendbufs * sizeof(MPI_Offset));
  memset(sendbuflen, 0, nsendbufs * sizeof(MPI_Offset));

  nrecvbufs = size; // TODO: if this is a non-aggregator, we could set this to 0...
  recvbuf = malloc(nrecvbufs * sizeof(char*));
  memset(recvbuf, 0, nrecvbufs * sizeof(char*));
  recvbuflen = malloc(size * sizeof(MPI_Offset));
  memset(recvbuflen, 0, nrecvbufs * sizeof(MPI_Offset));

  for(i = 0; i < stripecount * coratio; i++) {
    for(j = 0; j < size; j++) {

      if(rank == writers[i] && rank != j) {
        // Set up a async recv to gather data from process j to writer[i].

        for(k = 0; k < npieces[j]; k++) {
          if((lustreoffset[j][k] / stripesize) % (stripecount * coratio) == i) {
            recvbuflen[j] += lustrelength[j][k];
          }
        }

        if(recvbuflen[j] > 0) {
          recvbuf[j] = malloc(recvbuflen[j] * sizeof(char));
          MPI_Irecv(recvbuf[j], recvbuflen[j], MPI_BYTE,
                    j, 0, ncp->nciop->comm, &asyncreqs[reqcount++]);
        }

      } else if(rank == j && rank != writers[i]) {
        // Gather all data that will be sent from j > writer[i]

        for(k = 0; k < npieces[j]; k++) {
          if((lustreoffset[j][k] / stripesize) % (stripecount * coratio) == i) {
            sendbuflen[i] += lustrelength[j][k];
          }
        }

        if(sendbuflen[i] > 0) {
          sendbuf[i] = malloc(sendbuflen[i] * sizeof(char));
          sendbufoffset = 0;
          xbufoffset = 0;

          for(k = 0; k < npieces[j]; k++) {
            if((lustreoffset[j][k] / stripesize) % (stripecount * coratio) == i) {
              memcpy(sendbuf[i] + sendbufoffset, (char*)xbuf + xbufoffset, lustrelength[j][k]); 
              sendbufoffset += lustrelength[j][k];
            }
            
            xbufoffset += lustrelength[j][k];
          }     
        } 
      }
    }
  }

  // We want to ensure that all Irecvs have been posted before we do any Isends
  MPI_Barrier(ncp->nciop->comm);

  for(i = 0; i < stripecount * coratio; i++) {
    for(j = 0; j < size; j++) {

      if(rank == j && rank != writers[i]) {
        // Initiate a send from j > writers[i]

        if(sendbuf[i]) {
          MPI_Irsend(sendbuf[i], sendbuflen[i], MPI_BYTE, writers[i],
                     0, ncp->nciop->comm, &asyncreqs[reqcount++]);
        }

      } else if(rank == j && rank == writers[i]) {
        // No need to do a send from a process to itself, just copy it directly
        // to the stripe buffer.

        xbufoffset = 0;
        for(k = 0; k < npieces[j]; k++) {
          if((lustreoffset[j][k] / stripesize) % (stripecount * coratio) == i) {
            stripe = lustreoffset[j][k] / stripesize;
            if(0 == stripes[stripe]) {
              stripes[stripe] = malloc(stripesize * sizeof(char));
            }
            
            stripeoffset = lustreoffset[j][k] - (stripe * stripesize);
            memcpy(stripes[stripe] + stripeoffset, 
                   ((char*)xbuf) + xbufoffset, lustrelength[j][k]);
            
            localbytes += lustrelength[j][k];
            writebytes += lustrelength[j][k];
          }
          
          xbufoffset += lustrelength[j][k];
        }
      }
    }
  }

  MPI_Waitall(reqcount, asyncreqs, MPI_STATUSES_IGNORE);

  // We have now gathered all data at the appropriate aggregators, so we must
  // now copy from the recv buffers to the stripe buffers.
  for(i = 0; i < stripecount * coratio; i++) {
    for(j = 0; j < size; j++) {
      if(rank == writers[i] && rank != j && recvbuf[j]) {
        recvbufoffset = 0;
        for(k = 0; k < npieces[j]; k++) {
          if((lustreoffset[j][k] / stripesize) % (stripecount * coratio) == i) {
            stripe = lustreoffset[j][k] / stripesize;
            if(0 == stripes[stripe]) {
              stripes[stripe] = malloc(stripesize * sizeof(char));
            }

            stripeoffset = lustreoffset[j][k] - (stripe * stripesize);
            memcpy(stripes[stripe] + stripeoffset, recvbuf[j] + recvbufoffset, lustrelength[j][k]);
            recvbufoffset += lustrelength[j][k];
          }
        }
      }
    }
  }

  for(i = 0; i < nsendbufs; i++) {
    if(sendbuf[i])
      free(sendbuf[i]);
  }

  free(sendbuf);
  free(sendbuflen);

  for(i = 0; i < nrecvbufs; i++) {
    if(recvbuf[i])
      free(recvbuf[i]);
  }

  free(recvbuf);
  free(recvbuflen);

#ifdef WRITE_DEBUG_MESSAGES
  printf("Rank %03d: localbytes=%d, writebytes=%d\n", rank, (int)localbytes, (int)writebytes);
#endif

  return retval;
}

int Write(NC* ncp, NC_var* varp,
          MPI_File fh, char* stripes[], int nstripes, 
          int stripesize, int stripecount) {
  int retval = NC_NOERR;

  int i, rank;
  MPI_Offset stripeoffset, offset, length;
  MPI_Offset varoffset, varlength;
  MPI_Status status;

  MPI_Comm_rank(ncp->nciop->comm, &rank);

  varoffset = varp->begin;
  varlength = varp->len;

  for(i = 0; i < nstripes; i++) {
    if(stripes[i]) {
      
      stripeoffset =                            // Must cast the inputs, since
        (MPI_Offset)i * (MPI_Offset)stripesize; // result may be too big for an int.
      offset = 0;
      length = stripesize;
      
      if(stripeoffset < varoffset) {
        // If this is the first stripe we are writing, then it may not align
        // with the beginning of the variable, so we need to take care not
        // to overwrite data preceding the variable.

        offset = varoffset - stripeoffset;
        length -= offset;
      }
      
      if(stripeoffset + stripesize > varoffset + varlength) {
        // If this is the last stripe we are writing, then its end may not
        // align with the end of the variable, so we need to take care not
        // to overwrite the data following the variable.
        
        length -= (stripeoffset + stripesize) - 
          (varoffset + varlength);
      }

#ifdef WRITE_DEBUG_MESSAGES
      printf("Rank %03d: Writing to stripe %d:\n"
             "          stripeoffset=%d, offset=%d, length=%d\n",
             rank, i, (int)stripeoffset, (int)offset, (int)length);
#endif
      
      // TODO: check return codes
      MPI_File_seek(fh, stripeoffset + offset, MPI_SEEK_SET);
      MPI_File_write(fh, stripes[i] + offset, length, MPI_BYTE, &status);
    }
  }

  return retval;
}
