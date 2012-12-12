/**
   pism.h - Created by Timothy Morey on 11/28/2012

   This file contains definitions for the PISM optimizations in the parallel-
   netcdf library.
 */

#ifndef __PISM_H
#define __PISM_H

#include "nc.h"

int DoLustreOptimizedWrite(NC* nc, NC_var* varp, 
                           MPI_Offset start[], MPI_Offset count[],
                           void* xbuf, MPI_File fh);

int DataSpaceToVarSpace(NC* ncp, NC_var* varp, 
                        const MPI_Offset start[],
                        const MPI_Offset count[],
                        MPI_Offset varoffset[],
                        MPI_Offset varlength[],
                        int* len);

int VarSpaceToFileSpace(NC* ncp, NC_var* varp,
                        const MPI_Offset varoffset[], 
                        const MPI_Offset varlength[],
                        int nvarpieces,
                        MPI_Offset fileoffset[],
                        MPI_Offset filelength[],
                        int* nfilepieces);

int FileSpaceToLustreSpace(NC* ncp, NC_var* varp,
                           MPI_Offset stripesize, MPI_Offset stripecount,
                           const MPI_Offset fileoffset[],
                           const MPI_Offset filelength[],
                           int n_filepieces,
                           MPI_Offset lustreoffset[],
                           MPI_Offset lustrelength[],
                           int* nlustrepieces);

int SelectWriters(MPI_Comm comm, int stripecount, int writers[]);

int Redistribute(NC* ncp, NC_var* varp,
                 int stripesize, int stripecount,
                 const MPI_Offset** lustreoffset,
                 const MPI_Offset** lustrelength,
                 int npieces[],
                 void* xbuf,
                 int writers[],
                 char* stripes[], int nstripes);

int Write(NC* ncp, NC_var* varp,
          MPI_File fh, char* stripes[], int nstripes,
          int stripesize, int stripecount);

#endif
