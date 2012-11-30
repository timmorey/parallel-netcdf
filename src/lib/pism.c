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

int DoLustreOptimizedWrite(NC* ncp, NC_var* varp, 
                           const MPI_Offset start[], 
                           const MPI_Offset count[],
                           void* xbuf, MPI_File fh) {
  int retval = NC_NOERR;

  MPI_Offset varoffset[2048], varlength[2048];
  int nvarpieces = 2048;
  
  MPI_Offset fileoffset[2048], filelength[2048];
  int nfilepieces = 2048;
  
  MPI_Offset lustreoffset[4096], lustrelength[4096];
  int nlustrepieces = 4096;
  
  int rank;
  int i;
        
  MPI_Datatype view;
  MPI_Offset xbuf_offset;

  MPI_Info info;
  char hintvalue[MPI_MAX_INFO_VAL];
  int hintfound;
  
  MPI_Offset stripesize = 1048576;
  MPI_Offset stripecount = 4;

  int* writers;
  char** stripes;
  int nstripes;

  double starttime, initend, mapend, redistend, writeend;

  starttime = MPI_Wtime();

  MPI_Comm_rank(ncp->nciop->comm, &rank);

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

#ifdef WRITE_DEBUG_MESSAGES
  printf("Rank %03d: Collective write for '%s'\n", rank, varp->name->cp);

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

  writers = (int*)malloc(stripecount * sizeof(int));
  SelectWriters(ncp->nciop->comm, stripecount, writers);

  // TODO: how do we calculate the total number of stripes?
  nstripes = 16384;
  stripes = (char**)malloc(nstripes * sizeof(char*));
  memset(stripes, 0, nstripes * sizeof(char*));

  initend = MPI_Wtime();

  if(NC_NOERR != 
     (retval = DataSpaceToVarSpace(ncp, varp, start, count, 
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
                                      lustreoffset, lustrelength, &nlustrepieces))) {
    fprintf(stderr, "Rank %03d: FileSpaceToLustreSpace failed.\n", rank);
  }

  mapend = MPI_Wtime();
  
  if(NC_NOERR !=
     (retval = Redistribute(ncp, varp, 
                            stripesize, stripecount,
                            lustreoffset, lustrelength, nlustrepieces, xbuf, 
                            writers, stripes, nstripes))) {
    fprintf(stderr, "Rank %03d: Redistribute failed.\n", rank);
  }

  redistend = MPI_Wtime();

  if(NC_NOERR !=
     (retval = Write(ncp, varp, fh, 
                     stripes, nstripes, stripesize, stripecount))) {
    fprintf(stderr, "Rank %03d: Write failed.\n", rank);
  }

  writeend = MPI_Wtime();

  printf("Rank %03d (%s): init-time   = %8.4f s\n", 
         rank, varp->name->cp, initend - starttime);
  printf("Rank %03d (%s): map-time    = %8.4f s\n", 
         rank, varp->name->cp, mapend - initend);
  printf("Rank %03d (%s): redist-time = %8.4f s\n", 
         rank, varp->name->cp, redistend - mapend);
  printf("Rank %03d (%s): write-time  = %8.4f s\n", 
         rank, varp->name->cp, writeend - redistend);

  for(i = 0; i < nstripes; i++) {
    if(stripes[i])
      free(stripes[i]);
  }

  free(stripes);
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
  int rank;

  MPI_Comm_rank(ncp->nciop->comm, &rank);

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

#ifdef WRITE_DEBUG_MESSAGES
  printf("Rank %03d: Var space consists of %d contiguous regions:\n", 
         rank, *len);
  printf("Rank %03d:   %-18s  %-18s\n", rank, "Offset", "Length");
  printf("Rank %03d:   %-18s  %-18s\n", rank, "------", "------");
  for(i = 0; i < *len; i++) {
    printf("Rank %03d:   0x%016x  0x%016x\n", 
           rank, (int)varoffset[i], (int)varlength[i]);
  }
#endif

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
  int rank;

  /* TODO: For record variables, we always assume we are writing record 0 */

  MPI_Comm_rank(ncp->nciop->comm, &rank);

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
  
#ifdef WRITE_DEBUG_MESSAGES
  printf("Rank %03d: File space consists of %d contiguous regions:\n", 
         rank, *nfilepieces);
  printf("Rank %03d:   %-18s  %-18s\n", rank, "Offset", "Length");
  printf("Rank %03d:   %-18s  %-18s\n", rank, "------", "------");
  for(i = 0; i < *nfilepieces; i++) {
    printf("Rank %03d:   0x%016x  0x%016x\n", 
           rank, (int)fileoffset[i], (int)filelength[i]);
  }
#endif

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
  int rank;

  MPI_Comm_rank(ncp->nciop->comm, &rank);

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
  
#ifdef WRITE_DEBUG_MESSAGES
  printf("Rank %03d: Lustre space consists of %d contiguous regions:\n", 
         rank, *nlustrepieces);
  printf("Rank %03d:   %-18s  %-18s\n", rank, "Offset", "Length");
  printf("Rank %03d:   %-18s  %-18s\n", rank, "------", "------");
  for(i = 0; i < *nlustrepieces; i++) {
    printf("Rank %03d:   0x%016x  0x%016x\n", 
           rank, (int)lustreoffset[i], (int)lustrelength[i]);
  }
#endif

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
                 int stripesize, int stripecount,
                 const MPI_Offset lustreoffset[],
                 const MPI_Offset lustrelength[],
                 int npieces,
                 void* xbuf,
                 int writers[],
                 char* stripes[],
                 int nstripes) {
  int retval = 0;
  int rank, size;
  int i, j, k;
  int count;
  MPI_Offset offset, length;
  int stripe;
  MPI_Offset xbufoffset, stripeoffset;
  MPI_Status status;
  MPI_Datatype view;
  int unlimdim;
  MPI_Offset varoffset, varlength;
  MPI_Offset localbytes, writebytes;

  MPI_Comm_rank(ncp->nciop->comm, &rank);
  MPI_Comm_size(ncp->nciop->comm, &size);

  unlimdim = ncmpii_find_NC_Udim(&ncp->dims, NULL);
  varoffset = varp->begin;
  varlength = varp->xsz;
  for(i = 0; i < varp->ndims; i++) {
    if(varp->dimids[i] != unlimdim) {
      varlength *= varp->shape[i];
    }
  }

  localbytes = 0;
  writebytes = 0;

  // 1) Figure out which pieces writer[i] already has
  for(i = 0; i < stripecount; i++) {
    if(rank == writers[i]) {
      //printf("Rank %03d: gathering local data...\n", rank);
      xbufoffset = 0;
      for(k = 0; k < npieces; k++) {
        if((lustreoffset[k] / stripesize) % stripecount == i) {
          stripe = lustreoffset[k] / stripesize;
          if(0 == stripes[stripe]) {
            stripes[stripe] = (char*)malloc(stripesize * sizeof(char));
          }

          stripeoffset = lustreoffset[k] - (stripe * stripesize);
          memcpy(stripes[stripe] + stripeoffset, 
                 ((char*)xbuf) + xbufoffset, lustrelength[k]);

          localbytes += lustrelength[k];
          writebytes += lustrelength[k];
        }

        xbufoffset += lustrelength[k];
      }
    }
  }

  // 2) Send stripe fragments to the appropriate writers.
  for(i = 0; i < stripecount; i++) {
    for(j = 0; j < size; j++) {

      // No point sending data from one process to itself
      if(writers[i] == j) continue;

      if(rank == writers[i]) {
        // Then this process must receive data from process[j]

        // 1) Receive the number of pieces that process[j] will send
        MPI_Recv(&count, 1, MPI_INT, j, 0, ncp->nciop->comm, &status);

        // 2) Loop and receive the pieces
        for(k = 0; k < count; k++) {

          // 1) Recieve the offset and length
          MPI_Recv(&offset, 1, MPI_UNSIGNED_LONG, j, 0, ncp->nciop->comm, &status);
          MPI_Recv(&length, 1, MPI_UNSIGNED_LONG, j, 0, ncp->nciop->comm, &status);

          writebytes += length;

          // 2) Receive the data
          stripe = offset / stripesize;
          if(0 == stripes[stripe]) {
            stripes[stripe] = (char*)malloc(stripesize * sizeof(char));
          }

          stripeoffset = offset - (stripe * stripesize);
          MPI_Recv(stripes[stripe] + stripeoffset, length, MPI_BYTE, 
                   j, 0, ncp->nciop->comm, &status);
        }

      } else if(rank == j) {
        // Then this process must send data to writer[i]

        // 1) Count how many pieces I have for writer[i]
        count = 0;
        for(k = 0; k < npieces; k++) {
          if((lustreoffset[k] / stripesize) % stripecount == i) {
            count++;
          }
        }

        // 2) Send the count to writer[i]
        MPI_Send(&count, 1, MPI_INT, writers[i], 0, ncp->nciop->comm);

        // 3) Loop and send the pieces
        xbufoffset = 0;
        for(k = 0; k < npieces; k++) {
          if((lustreoffset[k] / stripesize) % stripecount == i) {
            // 1) Send the offset and length of the piece
            MPI_Send((void*)&lustreoffset[k], 1, MPI_UNSIGNED_LONG, 
                     writers[i], 0, ncp->nciop->comm);
            MPI_Send((void*)&lustrelength[k], 1, MPI_UNSIGNED_LONG, 
                     writers[i], 0, ncp->nciop->comm);

            // 2) Send the data for the piece
            MPI_Send(((char*)xbuf) + xbufoffset, lustrelength[k], MPI_BYTE, 
                     writers[i], 0, ncp->nciop->comm);
          }

          xbufoffset += lustrelength[k];
        }
      }
    }
  }

  printf("Rank %03d: localbytes=%d, writebytes=%d\n", rank, (int)localbytes, (int)writebytes);

  return retval;
}

int Write(NC* ncp, NC_var* varp,
          MPI_File fh, char* stripes[], int nstripes, 
          int stripesize, int stripecount) {
  int retval = NC_NOERR;

  int i, rank, unlimdim;
  MPI_Offset stripeoffset, offset, length;
  MPI_Offset varoffset, varlength;
  MPI_Status status;

  MPI_Comm_rank(ncp->nciop->comm, &rank);

  unlimdim = ncmpii_find_NC_Udim(&ncp->dims, NULL);
  varoffset = varp->begin;
  varlength = varp->xsz;
  for(i = 0; i < varp->ndims; i++) {
    if(varp->dimids[i] != unlimdim) {
      varlength *= varp->shape[i];
    }
  }

  for(i = 0; i < nstripes; i++) {
    if(stripes[i]) {
      
      stripeoffset = i * stripesize;
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
      
      // TODO: check return codes
      MPI_File_seek(fh, stripeoffset + offset, MPI_SEEK_SET);
      MPI_File_write(fh, stripes[i] + offset, length, MPI_BYTE, &status);
    }
  }

  return retval;
}
