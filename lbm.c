/*
** code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the bhatnagar-gross-krook collision step.
**
** the 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** a 2d grid:
**
**           cols
**       --- --- ---
**      | d | e | f |
** rows  --- --- ---
**      | a | b | c |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1d array:
**
**  --- --- --- --- --- ---
** | a | b | c | d | e | f |
**  --- --- --- --- --- ---
**
** grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./lbm -a av_vels.dat -f final_state.dat -p ../inputs/box.params
**
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "lbm.h"

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
    char * final_state_file = NULL;
    char * av_vels_file = NULL;
    char * param_file = NULL;

    accel_area_t accel_area;

    param_t  params;              /* struct to hold parameter values */
    float* cells     = NULL;    /* grid containing fluid densities */
    float* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float*  av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */

    int    ii;                    /*  generic counter */
    struct timeval timstr;        /* structure to hold elapsed time */
    struct rusage ru;             /* structure to hold CPU time--system and user */
    double tic,toc;               /* floating point numbers to calculate elapsed wallclock time */
    double usrtim;                /* floating point number to record elapsed user CPU time */
    double systim;                /* floating point number to record elapsed system CPU time */

    int device_id;
    lbm_context_t lbm_context;

    parse_args(argc, argv, &final_state_file, &av_vels_file, &param_file, &device_id);

    initialise(param_file, &accel_area, &params, &cells, &tmp_cells, &obstacles, &av_vels);
    
    opencl_initialise(device_id, params, accel_area, &lbm_context, cells, tmp_cells, obstacles);
    
   /* iterate for max_iters timesteps */
    gettimeofday(&timstr,NULL);
    tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);


      //Run kernel with auto work group sizes
      const size_t global[2] = {params.ny, params.nx};
      const size_t global2 = (accel_area.col_or_row == ACCEL_COLUMN) ? params.ny : params.nx;


      const int GLOBALSIZE = params.nx * params.ny;
      const int GROUPSIZE = GLOBALSIZE/LOCALSIZE;
      const size_t g = GLOBALSIZE;
      const size_t l = LOCALSIZE;
      const size_t local[2] = {1, LOCALSIZE};


    cl_int err;
    // kernel arguments
      err = clSetKernelArg(lbm_context.k_flow, 0, sizeof(param_t), &params);
      err |= clSetKernelArg(lbm_context.k_flow, 1, sizeof(accel_area_t), &accel_area);
      err |= clSetKernelArg(lbm_context.k_flow, 2, sizeof(cl_mem), &lbm_context.d_cells);
      err |= clSetKernelArg(lbm_context.k_flow, 3, sizeof(cl_mem), &lbm_context.d_obstacles);

      err |= clSetKernelArg(lbm_context.k_propagate, 0, sizeof(param_t), &params);
      err |= clSetKernelArg(lbm_context.k_propagate, 1, sizeof(cl_mem), &lbm_context.d_cells);
      err |= clSetKernelArg(lbm_context.k_propagate, 2, sizeof(cl_mem), &lbm_context.d_tmp_cells);
      /*
      err |= clSetKernelArg(lbm_context.k_rebound, 0, sizeof(param_t), &params);
      err |= clSetKernelArg(lbm_context.k_rebound, 1, sizeof(cl_mem), &lbm_context.d_cells);
      err |= clSetKernelArg(lbm_context.k_rebound, 2, sizeof(cl_mem), &lbm_context.d_tmp_cells);
      err |= clSetKernelArg(lbm_context.k_rebound, 3, sizeof(cl_mem), &lbm_context.d_obstacles);
*/      
      err |= clSetKernelArg(lbm_context.k_collision, 0, sizeof(param_t), &params);
      err |= clSetKernelArg(lbm_context.k_collision, 1, sizeof(cl_mem), &lbm_context.d_cells);
      err |= clSetKernelArg(lbm_context.k_collision, 2, sizeof(cl_mem), &lbm_context.d_tmp_cells);
      err |= clSetKernelArg(lbm_context.k_collision, 3, sizeof(cl_mem), &lbm_context.d_obstacles);

      err |= clSetKernelArg(lbm_context.k_velocity, 0, sizeof(param_t), &params);
      err |= clSetKernelArg(lbm_context.k_velocity, 1, sizeof(cl_mem), &lbm_context.d_cells);
      err |= clSetKernelArg(lbm_context.k_velocity, 2, sizeof(cl_mem), &lbm_context.d_obstacles);
      err |= clSetKernelArg(lbm_context.k_velocity, 3, sizeof(cl_float)*LOCALSIZE,NULL);
      err |= clSetKernelArg(lbm_context.k_velocity, 4, sizeof(cl_mem), &lbm_context.d_results);
     
      if (err != CL_SUCCESS)
	DIE("OpenCL error %d, could not set kernel arguments", err);


      int tot_cells = 0;
      for (ii = 0; ii < params.ny; ii++)
	{
	  for(int jj = 0; jj < params.nx; jj++)
	    {
	      if(!obstacles[ii*params.nx + jj])
		tot_cells ++;
	    }
	}

    for (ii = 0; ii < params.max_iters; ii++)
    {

      if(err != CL_SUCCESS)
	DIE("OpenCL error %d, could not write to buffer",err);
           

      err = clEnqueueNDRangeKernel(lbm_context.queue,lbm_context.k_flow,1,NULL,&global2,NULL,0, NULL, NULL);

      err |= clEnqueueNDRangeKernel(lbm_context.queue,lbm_context.k_propagate,2,NULL,global,local,0, NULL, NULL);

      err |= clEnqueueNDRangeKernel(lbm_context.queue,lbm_context.k_collision,2,NULL,global,local,0, NULL, NULL);

      err |= clEnqueueNDRangeKernel(lbm_context.queue,lbm_context.k_velocity,1,NULL,&g, &l,0,NULL,NULL);
      
     if(err != CL_SUCCESS)
	DIE("OpenCL error %d, could not run kernel",err);
      
     cl_float * results = (cl_float*) malloc(sizeof(cl_float)*(GROUPSIZE));
      err = clEnqueueReadBuffer(lbm_context.queue, lbm_context.d_results, CL_TRUE, 0, 
				      sizeof(cl_float)*(GROUPSIZE),results,0,NULL,NULL);

	    if(err != CL_SUCCESS)
	      DIE("OpenCL error %d, could not read buffer", err);

	    for(int x = 0; x < (GROUPSIZE); x ++)
	      {
		//if(results[x] > 0 || results[x] < 0)
		  //printf("results:[%d][%d], %.12E\n", ii, x, results[x]);
		av_vels[ii]+=results[x];
	      }
	    av_vels[ii] = av_vels[ii]/ (float) tot_cells;

	    free(results);
      
       #ifdef DEBUG
        printf("==timestep: %d==\n", ii);
        printf("av velocity: %.12E\n", av_vels[ii]);
        printf("tot density: %.12E\n", total_density(params, cells));
		#endif
    }

    err = clEnqueueReadBuffer(lbm_context.queue, lbm_context.d_cells, CL_TRUE, 0, 
      		sizeof(float)*(params.nx*params.ny*NSPEEDS),cells,0,NULL,NULL);
      	

    // Do not remove this, or the timing will be incorrect!
    clFinish(lbm_context.queue);

    gettimeofday(&timstr,NULL);
    toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr=ru.ru_utime;
    usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    timstr=ru.ru_stime;
    systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params,av_vels[params.max_iters-1]));
    printf("Elapsed time:\t\t\t%.6f (s)\n", toc-tic);
    printf("Elapsed user CPU time:\t\t%.6f (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6f (s)\n", systim);

    write_values(final_state_file, av_vels_file, params, cells, obstacles, av_vels);
    finalise(&cells, &tmp_cells, &obstacles, &av_vels);
    opencl_finalise(lbm_context);

    return EXIT_SUCCESS;
}

void write_values(const char * final_state_file, const char * av_vels_file,
    const param_t params, float* cells, int* obstacles, float* av_vels)
{
    FILE* fp;                     /* file pointer */
    int ii,jj,kk;                 /* generic counters */
    const float c_sq = 1.0/3.0;  /* sq. of speed of sound */
    float local_density;         /* per grid cell sum of densities */
    float pressure;              /* fluid pressure in grid cell */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(final_state_file, "w");

    if (fp == NULL)
    {
        DIE("could not open file output file");
    }

    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.nx; jj++)
        {
            /* an occupied cell */
            if (obstacles[ii*params.nx + jj])
            {
                u_x = u_y = u = 0.0;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else
            {
                local_density = 0.0;

                for (kk = 0; kk < NSPEEDS; kk++)
                {
		  local_density += cells[ii*params.nx + jj+kk*(params.nx*params.ny)];
                }

                /* compute x velocity component */
                u_x = (cells[ii*params.nx + jj+(params.nx*params.ny)] +
                        cells[ii*params.nx + jj+5*(params.nx*params.ny)] +
		       cells[ii*params.nx + jj+8*(params.nx*params.ny)]
		       - (cells[ii*params.nx + jj+3*(params.nx*params.ny)] +
			  cells[ii*params.nx + jj+6*(params.nx*params.ny)] +
			  cells[ii*params.nx + jj+7*(params.nx*params.ny)]))
                    / local_density;

                /* compute y velocity component */
                u_y = (cells[ii*params.nx + jj+2*(params.nx*params.ny)] +
		       cells[ii*params.nx + jj+5*(params.nx*params.ny)] +
		       cells[ii*params.nx + jj+6*(params.nx*params.ny)]
		       - (cells[ii*params.nx + jj+4*(params.nx*params.ny)] +
			  cells[ii*params.nx + jj+7*(params.nx*params.ny)] +
			  cells[ii*params.nx + jj+8*(params.nx*params.ny)]))
                    / local_density;

                /* compute norm of velocity */
                u = sqrt((u_x * u_x) + (u_y * u_y));

                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp,"%d %d %.12E %.12E %.12E %.12E %d\n",
                jj,ii,u_x,u_y,u,pressure,obstacles[ii*params.nx + jj]);
        }
    }

    fclose(fp);

    fp = fopen(av_vels_file, "w");
    if (fp == NULL)
    {
        DIE("could not open file output file");
    }

    for (ii = 0; ii < params.max_iters; ii++)
    {
        fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);
}

float calc_reynolds(const param_t params, float av_vel)
{
    const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

    return av_vel * params.reynolds_dim / viscosity;
}

float total_density(const param_t params, float* cells)
{
    int ii,jj,kk;        /* generic counters */
    float total = 0.0;  /* accumulator */

    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.ny; jj++)
        {
            for (kk = 0; kk < NSPEEDS; kk++)
            {
	      total += cells[ii*params.nx + jj*(kk+1)];            }
        }
    }

    return total;
}

