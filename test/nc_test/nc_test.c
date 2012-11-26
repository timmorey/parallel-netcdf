/*********************************************************************
 *   Copyright 1996, UCAR/Unidata
 *   See netcdf/COPYRIGHT file for copying and redistribution conditions.
 *   $Id: nc_test.c 1050 2012-05-11 06:07:39Z wkliao $
 *********************************************************************/

#include <unistd.h>
#include "tests.h"

/*
 * Test driver for netCDF-3 interface.  This program performs tests against
 * the netCDF-3 specification for all user-level functions in an
 * implementation of the netCDF library.
 *
 * Unless invoked with "-r" (read_only) option, must be invoked from a
 * directory in which the invoker has write permission.
 *
 * Files:
 * The read-only tests read files:
 *     test.nc (see below)
 *     test_get.c (used merely as an example of a non-netCDF file)
 * 
 * The write tests 
 *     read test.nc (see below) 
 *     write scratch.nc (deleted after each test)
 * 
 * The file test.nc is created by running ncmpi_test with the -c (create) option.
 * It is described by the following global variables.
 */


/*
 * global variables (defined by command line processing in main())
 * related use of CDF-1 vs CDF-2 file formats
 */
int use_cdf2 = 0; 	/* 0: use CDF-1, 1: use CDF-2 */
int extra_flags = 0; /* if using CDF-2 format, will be set to NC_64BIT_OFFSET */

/* 
 * global variables (defined by function init_gvars) describing file test.nc
 */
char dim_name[NDIMS][3];
MPI_Offset dim_len[NDIMS];
char var_name[NVARS][2+MAX_RANK];
nc_type var_type[NVARS];
size_t var_rank[NVARS];
int var_dimid[NVARS][MAX_RANK];
MPI_Offset var_shape[NVARS][MAX_RANK];
size_t var_nels[NVARS];
size_t var_natts[NVARS];
char att_name[NVARS][MAX_NATTS][2];
char gatt_name[NGATTS][3];
nc_type att_type[NVARS][NGATTS];
nc_type gatt_type[NGATTS];
size_t att_len[NVARS][MAX_NATTS];
size_t gatt_len[NGATTS];

/* 
 * command-line options
 */
static int  create_file;	/* if 1, create file test.nc */
int  read_only;		/* if 1, don't try to change files */
int  verbose;		/* if 1, print details of tests */
int  max_nmpt;		/* max. number of messages per test */

/* 
 * Misc. global variables
 */
int  nfails;		/* number of failures in specific test */
static char *progname;
char testfile[] = "test.nc";    /* read-only testfile */
char scratch[] = "scratch.nc";  /* writable scratch file */
MPI_Comm comm = MPI_COMM_WORLD; /* mpi communicator for parallel-netcdf */

static void
usage(void)
{
    error("%s [-hrv] [-n <MAX_NMPT>]\n", progname);
    error("   [-h] Print help\n" );
    error("   [-c] Create file test.nc (Do not do tests)\n" );
    error("   [-r] Just do read-only tests\n" );
    error("   [-v] Verbose mode\n" );
    error("   [-2] (with -c) create file with CDF-2 format\n" );
    error("   [-n <MAX_NMPT>] max. number of messages per test (Default: 8)\n");
}

#define NC_TEST(func) \
    print( "*** Testing %-25s ... ",#func);\
    nfails = 0;\
    test_ ## func();\
    nfailsTotal += nfails;\
    if (verbose) \
	print("\n"); \
    if ( nfails == 0) \
        print( "ok\n");\
    else\
        print( "\n\t### %d FAILURES TESTING %s! ###\n", nfails, #func)


#if 1		/* both CRAY MPP and OSF/1 Alpha systems need this */
#include <signal.h>
#endif /* T90 */

int
main(int argc, char *argv[])
{
    extern char *optarg;
    int c;
    int  nfailsTotal = 0;        /* total number of failures */

#if 1		/* both CRAY MPP and OSF/1 Alpha systems need this */
	/*
	 * Some of the extreme test assignments in this program trigger
         * floating point exceptions on CRAY T90
	 */
	(void) signal(SIGFPE, SIG_IGN);
#endif

    MPI_Init(&argc, &argv);

    progname = argv[0];
    create_file = 0;            /* file test.nc will normally already exist */
    read_only = 0;               /* assume may write in test dir as default */
    verbose = 0;
    max_nmpt = 8;
    while ((c = getopt(argc, argv, "c2hrvn:")) != EOF)
      switch(c) {
	case 'c':		/* Create file test.nc */
	  create_file = 1;
	  break;
	case 'r':		/* just perform read-only tests */
	  read_only = 1;
	  break;
	case 'v':		/* verbose mode */
	  verbose = 1;
	  break;
	case 'n':		/* verbose mode */
	  max_nmpt = atoi(optarg);
	  break;
	case '2':
	  use_cdf2 = 1;
	  extra_flags = NC_64BIT_OFFSET;
	  break;
	case 'h':
	case '?':
	  usage();
	  return 1;
      }

    /* Initialize global variables defining test file */
    init_gvars();

    if ( create_file ) {
	write_file(testfile);
	MPI_Finalize();
	return nfailsTotal > 0;
    }

    /* delete any existing scratch netCDF file */
    if ( ! read_only ) {
        if (access(scratch, F_OK) == 0)
            unlink(scratch);
    }

    /* Test read-only functions, using pregenerated test-file */
    NC_TEST(ncmpi_strerror);
    NC_TEST(ncmpi_open);
    NC_TEST(ncmpi_close);
    NC_TEST(ncmpi_inq);
    NC_TEST(ncmpi_inq_dimid);
    NC_TEST(ncmpi_inq_dim);
    NC_TEST(ncmpi_inq_dimlen);
    NC_TEST(ncmpi_inq_dimname);
    NC_TEST(ncmpi_inq_varid);
    NC_TEST(ncmpi_inq_var);
    NC_TEST(ncmpi_inq_natts);
    NC_TEST(ncmpi_inq_ndims);
    NC_TEST(ncmpi_inq_nvars);
    NC_TEST(ncmpi_inq_unlimdim);
    NC_TEST(ncmpi_inq_vardimid);
    NC_TEST(ncmpi_inq_varname);
    NC_TEST(ncmpi_inq_varnatts);
    NC_TEST(ncmpi_inq_varndims);
    NC_TEST(ncmpi_inq_vartype);
    NC_TEST(ncmpi_get_var_text);
    NC_TEST(ncmpi_get_var_schar);
    NC_TEST(ncmpi_get_var_uchar); 
    NC_TEST(ncmpi_get_var_short);
    NC_TEST(ncmpi_get_var_int);
    NC_TEST(ncmpi_get_var_long); 
    NC_TEST(ncmpi_get_var_float);
    NC_TEST(ncmpi_get_var_double);
    NC_TEST(ncmpi_get_var_ushort);
    NC_TEST(ncmpi_get_var_uint);
    NC_TEST(ncmpi_get_var_longlong);
    NC_TEST(ncmpi_get_var_ulonglong);
    NC_TEST(ncmpi_get_var1_text);
    NC_TEST(ncmpi_get_var1_schar);
    NC_TEST(ncmpi_get_var1_uchar); 
    NC_TEST(ncmpi_get_var1_short);
    NC_TEST(ncmpi_get_var1_int);
    NC_TEST(ncmpi_get_var1_long);
    NC_TEST(ncmpi_get_var1_float);
    NC_TEST(ncmpi_get_var1_double);
    NC_TEST(ncmpi_get_var1_ushort);
    NC_TEST(ncmpi_get_var1_uint);
    NC_TEST(ncmpi_get_var1_longlong);
    NC_TEST(ncmpi_get_var1_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_get_var1);
#endif /* TEST_VOIDSTAR */
    NC_TEST(ncmpi_get_vara_text);
    NC_TEST(ncmpi_get_vara_schar);
    NC_TEST(ncmpi_get_vara_uchar);
    NC_TEST(ncmpi_get_vara_short);
    NC_TEST(ncmpi_get_vara_int);
    NC_TEST(ncmpi_get_vara_long); 
    NC_TEST(ncmpi_get_vara_float);
    NC_TEST(ncmpi_get_vara_double);
    NC_TEST(ncmpi_get_vara_ushort);
    NC_TEST(ncmpi_get_vara_uint);
    NC_TEST(ncmpi_get_vara_longlong);
    NC_TEST(ncmpi_get_vara_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_get_vara);
#endif /* TEST_VOIDSTAR */
    NC_TEST(ncmpi_get_vars_text);
    NC_TEST(ncmpi_get_vars_schar);
    NC_TEST(ncmpi_get_vars_uchar);
    NC_TEST(ncmpi_get_vars_short);
    NC_TEST(ncmpi_get_vars_int);
    NC_TEST(ncmpi_get_vars_long); 
    NC_TEST(ncmpi_get_vars_float);
    NC_TEST(ncmpi_get_vars_double);
    NC_TEST(ncmpi_get_vars_ushort);
    NC_TEST(ncmpi_get_vars_uint);
    NC_TEST(ncmpi_get_vars_longlong);
    NC_TEST(ncmpi_get_vars_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_get_vars);
#endif /* TEST_VOIDSTAR */
    NC_TEST(ncmpi_get_varm_text);
    NC_TEST(ncmpi_get_varm_schar);
    NC_TEST(ncmpi_get_varm_uchar);
    NC_TEST(ncmpi_get_varm_short);
    NC_TEST(ncmpi_get_varm_int);
    NC_TEST(ncmpi_get_varm_long);
    NC_TEST(ncmpi_get_varm_float);
    NC_TEST(ncmpi_get_varm_double);
    NC_TEST(ncmpi_get_varm_ushort);
    NC_TEST(ncmpi_get_varm_uint);
    NC_TEST(ncmpi_get_varm_longlong);
    NC_TEST(ncmpi_get_varm_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_get_varm);
#endif /* TEST_VOIDSTAR */
    NC_TEST(ncmpi_get_att_text);
    NC_TEST(ncmpi_get_att_schar);
    NC_TEST(ncmpi_get_att_uchar);
    NC_TEST(ncmpi_get_att_short);
    NC_TEST(ncmpi_get_att_int);
    NC_TEST(ncmpi_get_att_long);
    NC_TEST(ncmpi_get_att_float);
    NC_TEST(ncmpi_get_att_double);
    NC_TEST(ncmpi_get_att_ushort);
    NC_TEST(ncmpi_get_att_uint);
    NC_TEST(ncmpi_get_att_longlong);
    NC_TEST(ncmpi_get_att_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_get_att);
#endif /* TEST_VOIDSTAR */
    NC_TEST(ncmpi_inq_att);
    NC_TEST(ncmpi_inq_attname);
    NC_TEST(ncmpi_inq_attid);
    NC_TEST(ncmpi_inq_attlen);
    NC_TEST(ncmpi_inq_atttype);

    /* nonblocking I/O */
    NC_TEST(ncmpi_iget_var_text);
    NC_TEST(ncmpi_iget_var_schar);
    NC_TEST(ncmpi_iget_var_uchar); 
    NC_TEST(ncmpi_iget_var_short);
    NC_TEST(ncmpi_iget_var_int);
    NC_TEST(ncmpi_iget_var_long); 
    NC_TEST(ncmpi_iget_var_float);
    NC_TEST(ncmpi_iget_var_double);
    NC_TEST(ncmpi_iget_var_ushort);
    NC_TEST(ncmpi_iget_var_uint);
    NC_TEST(ncmpi_iget_var_longlong);
    NC_TEST(ncmpi_iget_var_ulonglong);
    NC_TEST(ncmpi_iget_var1_text);
    NC_TEST(ncmpi_iget_var1_schar);
    NC_TEST(ncmpi_iget_var1_uchar); 
    NC_TEST(ncmpi_iget_var1_short);
    NC_TEST(ncmpi_iget_var1_int);
    NC_TEST(ncmpi_iget_var1_long);
    NC_TEST(ncmpi_iget_var1_float);
    NC_TEST(ncmpi_iget_var1_double);
    NC_TEST(ncmpi_iget_var1_ushort);
    NC_TEST(ncmpi_iget_var1_uint);
    NC_TEST(ncmpi_iget_var1_longlong);
    NC_TEST(ncmpi_iget_var1_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_iget_var1);
#endif /* TEST_VOIDSTAR */
    NC_TEST(ncmpi_iget_vara_text);
    NC_TEST(ncmpi_iget_vara_schar);
    NC_TEST(ncmpi_iget_vara_uchar);
    NC_TEST(ncmpi_iget_vara_short);
    NC_TEST(ncmpi_iget_vara_int);
    NC_TEST(ncmpi_iget_vara_long); 
    NC_TEST(ncmpi_iget_vara_float);
    NC_TEST(ncmpi_iget_vara_double);
    NC_TEST(ncmpi_iget_vara_ushort);
    NC_TEST(ncmpi_iget_vara_uint);
    NC_TEST(ncmpi_iget_vara_longlong);
    NC_TEST(ncmpi_iget_vara_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_iget_vara);
#endif /* TEST_VOIDSTAR */
    NC_TEST(ncmpi_iget_vars_text);
    NC_TEST(ncmpi_iget_vars_schar);
    NC_TEST(ncmpi_iget_vars_uchar);
    NC_TEST(ncmpi_iget_vars_short);
    NC_TEST(ncmpi_iget_vars_int);
    NC_TEST(ncmpi_iget_vars_long); 
    NC_TEST(ncmpi_iget_vars_float);
    NC_TEST(ncmpi_iget_vars_double);
    NC_TEST(ncmpi_iget_vars_ushort);
    NC_TEST(ncmpi_iget_vars_uint);
    NC_TEST(ncmpi_iget_vars_longlong);
    NC_TEST(ncmpi_iget_vars_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_iget_vars);
#endif /* TEST_VOIDSTAR */
    NC_TEST(ncmpi_iget_varm_text);
    NC_TEST(ncmpi_iget_varm_schar);
    NC_TEST(ncmpi_iget_varm_uchar);
    NC_TEST(ncmpi_iget_varm_short);
    NC_TEST(ncmpi_iget_varm_int);
    NC_TEST(ncmpi_iget_varm_long);
    NC_TEST(ncmpi_iget_varm_float);
    NC_TEST(ncmpi_iget_varm_double);
    NC_TEST(ncmpi_iget_varm_ushort);
    NC_TEST(ncmpi_iget_varm_uint);
    NC_TEST(ncmpi_iget_varm_longlong);
    NC_TEST(ncmpi_iget_varm_ulonglong);
#ifdef TEST_VOIDSTAR
    NC_TEST(ncmpi_iget_varm);
#endif /* TEST_VOIDSTAR */

	/* Test write functions */
    if (! read_only) {
	NC_TEST(ncmpi_create);
	NC_TEST(ncmpi_redef);
	/* NC_TEST(ncmpi_enddef); *//* redundant */
	NC_TEST(ncmpi_sync);
	NC_TEST(ncmpi_abort);
	NC_TEST(ncmpi_def_dim);
	NC_TEST(ncmpi_rename_dim);
	NC_TEST(ncmpi_def_var);
	NC_TEST(ncmpi_put_var_text);
	NC_TEST(ncmpi_put_var_schar);
	NC_TEST(ncmpi_put_var_uchar);
	NC_TEST(ncmpi_put_var_short);
	NC_TEST(ncmpi_put_var_int);
	NC_TEST(ncmpi_put_var_long); 
	NC_TEST(ncmpi_put_var_float);
	NC_TEST(ncmpi_put_var_double);
	NC_TEST(ncmpi_put_var_ushort);
	NC_TEST(ncmpi_put_var_uint);
	NC_TEST(ncmpi_put_var_longlong);
	NC_TEST(ncmpi_put_var_ulonglong);
	NC_TEST(ncmpi_put_var1_text);
	NC_TEST(ncmpi_put_var1_schar);
	NC_TEST(ncmpi_put_var1_uchar); 
	NC_TEST(ncmpi_put_var1_short);
	NC_TEST(ncmpi_put_var1_int);
	NC_TEST(ncmpi_put_var1_long); 
	NC_TEST(ncmpi_put_var1_float);
	NC_TEST(ncmpi_put_var1_double);
	NC_TEST(ncmpi_put_var1_ushort);
	NC_TEST(ncmpi_put_var1_uint);
	NC_TEST(ncmpi_put_var1_longlong);
	NC_TEST(ncmpi_put_var1_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_put_var1);
#endif /* TEST_VOIDSTAR */
	NC_TEST(ncmpi_put_vara_text);
	NC_TEST(ncmpi_put_vara_schar); 
	NC_TEST(ncmpi_put_vara_uchar);
	NC_TEST(ncmpi_put_vara_short);
	NC_TEST(ncmpi_put_vara_int);
	NC_TEST(ncmpi_put_vara_long);
	NC_TEST(ncmpi_put_vara_float);
	NC_TEST(ncmpi_put_vara_double);
	NC_TEST(ncmpi_put_vara_ushort);
	NC_TEST(ncmpi_put_vara_uint);
	NC_TEST(ncmpi_put_vara_longlong);
	NC_TEST(ncmpi_put_vara_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_put_vara);
#endif /* TEST_VOIDSTAR */
	NC_TEST(ncmpi_put_vars_text);
	NC_TEST(ncmpi_put_vars_schar);
	NC_TEST(ncmpi_put_vars_uchar);
	NC_TEST(ncmpi_put_vars_short);
	NC_TEST(ncmpi_put_vars_int);
	NC_TEST(ncmpi_put_vars_long); 
	NC_TEST(ncmpi_put_vars_float);
	NC_TEST(ncmpi_put_vars_double);
	NC_TEST(ncmpi_put_vars_ushort);
	NC_TEST(ncmpi_put_vars_uint);
	NC_TEST(ncmpi_put_vars_longlong);
	NC_TEST(ncmpi_put_vars_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_put_vars);
#endif /* TEST_VOIDSTAR */
	NC_TEST(ncmpi_put_varm_text);
	NC_TEST(ncmpi_put_varm_schar);
	NC_TEST(ncmpi_put_varm_uchar);
	NC_TEST(ncmpi_put_varm_short);
	NC_TEST(ncmpi_put_varm_int);
	NC_TEST(ncmpi_put_varm_long);
	NC_TEST(ncmpi_put_varm_float);
	NC_TEST(ncmpi_put_varm_double);
	NC_TEST(ncmpi_put_varm_ushort);
	NC_TEST(ncmpi_put_varm_uint);
	NC_TEST(ncmpi_put_varm_longlong);
	NC_TEST(ncmpi_put_varm_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_put_varm);
#endif /* TEST_VOIDSTAR */
	NC_TEST(ncmpi_rename_var);
	NC_TEST(ncmpi_put_att_text);
	NC_TEST(ncmpi_put_att_schar);
	NC_TEST(ncmpi_put_att_uchar);
	NC_TEST(ncmpi_put_att_short);
	NC_TEST(ncmpi_put_att_int);
	NC_TEST(ncmpi_put_att_long);
	NC_TEST(ncmpi_put_att_float);
	NC_TEST(ncmpi_put_att_double);
	NC_TEST(ncmpi_put_att_ushort);
	NC_TEST(ncmpi_put_att_uint);
	NC_TEST(ncmpi_put_att_longlong);
	NC_TEST(ncmpi_put_att_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_put_att);
#endif /* TEST_VOIDSTAR */
	NC_TEST(ncmpi_copy_att);
	NC_TEST(ncmpi_rename_att);
	NC_TEST(ncmpi_del_att);
	NC_TEST(ncmpi_set_fill);

	NC_TEST(ncmpi_delete);

        /* test nonblocking APIs */
	NC_TEST(ncmpi_iput_var_text);
	NC_TEST(ncmpi_iput_var_schar);
	NC_TEST(ncmpi_iput_var_uchar);
	NC_TEST(ncmpi_iput_var_short);
	NC_TEST(ncmpi_iput_var_int);
	NC_TEST(ncmpi_iput_var_long); 
	NC_TEST(ncmpi_iput_var_float);
	NC_TEST(ncmpi_iput_var_double);
	NC_TEST(ncmpi_iput_var_ushort);
	NC_TEST(ncmpi_iput_var_uint);
	NC_TEST(ncmpi_iput_var_longlong);
	NC_TEST(ncmpi_iput_var_ulonglong);
	NC_TEST(ncmpi_iput_var1_text);
	NC_TEST(ncmpi_iput_var1_schar);
	NC_TEST(ncmpi_iput_var1_uchar); 
	NC_TEST(ncmpi_iput_var1_short);
	NC_TEST(ncmpi_iput_var1_int);
	NC_TEST(ncmpi_iput_var1_long); 
	NC_TEST(ncmpi_iput_var1_float);
	NC_TEST(ncmpi_iput_var1_double);
	NC_TEST(ncmpi_iput_var1_ushort);
	NC_TEST(ncmpi_iput_var1_uint);
	NC_TEST(ncmpi_iput_var1_longlong);
	NC_TEST(ncmpi_iput_var1_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_iput_var1);
#endif /* TEST_VOIDSTAR */
	NC_TEST(ncmpi_iput_vara_text);
	NC_TEST(ncmpi_iput_vara_schar); 
	NC_TEST(ncmpi_iput_vara_uchar);
	NC_TEST(ncmpi_iput_vara_short);
	NC_TEST(ncmpi_iput_vara_int);
	NC_TEST(ncmpi_iput_vara_long);
	NC_TEST(ncmpi_iput_vara_float);
	NC_TEST(ncmpi_iput_vara_double);
	NC_TEST(ncmpi_iput_vara_ushort);
	NC_TEST(ncmpi_iput_vara_uint);
	NC_TEST(ncmpi_iput_vara_longlong);
	NC_TEST(ncmpi_iput_vara_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_iput_vara);
#endif /* TEST_VOIDSTAR */
	NC_TEST(ncmpi_iput_vars_text);
	NC_TEST(ncmpi_iput_vars_schar);
	NC_TEST(ncmpi_iput_vars_uchar);
	NC_TEST(ncmpi_iput_vars_short);
	NC_TEST(ncmpi_iput_vars_int);
	NC_TEST(ncmpi_iput_vars_long); 
	NC_TEST(ncmpi_iput_vars_float);
	NC_TEST(ncmpi_iput_vars_double);
	NC_TEST(ncmpi_iput_vars_ushort);
	NC_TEST(ncmpi_iput_vars_uint);
	NC_TEST(ncmpi_iput_vars_longlong);
	NC_TEST(ncmpi_iput_vars_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_iput_vars);
#endif /* TEST_VOIDSTAR */
	NC_TEST(ncmpi_iput_varm_text);
	NC_TEST(ncmpi_iput_varm_schar);
	NC_TEST(ncmpi_iput_varm_uchar);
	NC_TEST(ncmpi_iput_varm_short);
	NC_TEST(ncmpi_iput_varm_int);
	NC_TEST(ncmpi_iput_varm_long);
	NC_TEST(ncmpi_iput_varm_float);
	NC_TEST(ncmpi_iput_varm_double);
	NC_TEST(ncmpi_iput_varm_ushort);
	NC_TEST(ncmpi_iput_varm_uint);
	NC_TEST(ncmpi_iput_varm_longlong);
	NC_TEST(ncmpi_iput_varm_ulonglong);
#ifdef TEST_VOIDSTAR
	NC_TEST(ncmpi_iput_varm);
#endif /* TEST_VOIDSTAR */
    }
    MPI_Finalize();

    print( "\nNOTE: parallel-netcdf expects to see 0 failures");
    print( "\nTotal number of failures: %d\n", nfailsTotal);
    if (nfailsTotal == 0) 
        return 0;
    else 
        return nfailsTotal > 0;
}
