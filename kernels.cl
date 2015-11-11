#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

/* struct to hold the 'speed' values */
typedef struct {
    double speeds[NSPEEDS];
} speed_t;

/* struct to hold the parameter values */
typedef struct {
    int nx;            /* no. of cells in x-direction */
    int ny;            /* no. of cells in y-direction */
    int max_iters;      /* no. of iterations */
    int reynolds_dim;  /* dimension for Reynolds number */
    double density;       /* density per link */
    double accel;         /* density redistribution */
    double omega;         /* relaxation parameter */
} param_t;

typedef enum { ACCEL_ROW=0, ACCEL_COLUMN=1 } accel_e;
typedef struct {
    int col_or_row;
    int idx;
} accel_area_t;


__kernel void accelerate_flow(const param_t params, const accel_area_t accel_area,
    __global speed_t* cells, __global int* obstacles)
{
    int ii,jj;     /* generic counters */
    
    double w1,w2;  /* weighting factors */
 /* compute weighting factors */
    w1 = params.density * params.accel / 9.0;
    w2 = params.density * params.accel / 36.0;

    if (accel_area.col_or_row == ACCEL_COLUMN)
    {
        jj = accel_area.idx;
	ii = get_global_id(0);
		/* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[ii*params.nx + jj].speeds[4] - w1) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[8] - w2) > 0.0 )
            {
	    /* increase 'north-side' densities */
                cells[ii*params.nx + jj].speeds[2] += w1;
                cells[ii*params.nx + jj].speeds[5] += w2;
                cells[ii*params.nx + jj].speeds[6] += w2;
		/* decrease 'south-side' densities */
                cells[ii*params.nx + jj].speeds[4] -= w1;
                cells[ii*params.nx + jj].speeds[7] -= w2;
                cells[ii*params.nx + jj].speeds[8] -= w2;
		}
	}
	else
    	{
        ii = accel_area.idx;
	jj = get_global_id(0);

	/* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
	     (cells[ii*params.nx + jj].speeds[3] - w1) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[6] - w2) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 )
            {
	    /* increase 'east-side' densities */
                cells[ii*params.nx + jj].speeds[1] += w1;
                cells[ii*params.nx + jj].speeds[5] += w2;
                cells[ii*params.nx + jj].speeds[8] += w2;
		/* decrease 'west-side' densities */
                cells[ii*params.nx + jj].speeds[3] -= w1;
                cells[ii*params.nx + jj].speeds[6] -= w2;
                cells[ii*params.nx + jj].speeds[7] -= w2;
		}
	}
}

__kernel void propagate(const param_t params, __global speed_t* cells, __global speed_t* tmp_cells)
{
	int ii = get_global_id(0);
	int jj = get_global_id(1);
            int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */
            /* determine indices of axis-direction neighbours
	    ** respecting periodic boundary conditions (wrap around) */
            y_n = (ii + 1) % params.ny;
            x_e = (jj + 1) % params.nx;
            y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
            x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

	    tmp_cells[ii *params.nx + jj].speeds[0]  = cells[ii*params.nx + jj].speeds[0]; /* central cell, */
                                                     /* no movement   */
            tmp_cells[ii *params.nx + x_e].speeds[1] = cells[ii*params.nx + jj].speeds[1]; /* east */
            tmp_cells[y_n*params.nx + jj].speeds[2]  = cells[ii*params.nx + jj].speeds[2]; /* north */
            tmp_cells[ii *params.nx + x_w].speeds[3] = cells[ii*params.nx + jj].speeds[3]; /* west */
            tmp_cells[y_s*params.nx + jj].speeds[4]  = cells[ii*params.nx + jj].speeds[4]; /* south */
	    tmp_cells[y_n*params.nx + x_e].speeds[5] = cells[ii*params.nx + jj].speeds[5]; /* north-east */
            tmp_cells[y_n*params.nx + x_w].speeds[6] = cells[ii*params.nx + jj].speeds[6]; /* north-west */
            tmp_cells[y_s*params.nx + x_w].speeds[7] = cells[ii*params.nx + jj].speeds[7]; /* south-west */
            tmp_cells[y_s*params.nx + x_e].speeds[8] = cells[ii*params.nx + jj].speeds[8]; /* south-east */

}

__kernel void rebound(const param_t params, __global speed_t* cells,
	      		     __global speed_t* tmp_cells, __global int* obstacles)
{
	int ii = get_global_id(0);
	int jj = get_global_id(1);

	if(obstacles[ii*params.nx + jj])
	{
		cells[ii*params.nx+jj].speeds[1] = tmp_cells[ii*params.nx+jj].speeds[3]; 
		cells[ii*params.nx+jj].speeds[2] = tmp_cells[ii*params.nx+jj].speeds[4]; 
		cells[ii*params.nx+jj].speeds[3] = tmp_cells[ii*params.nx+jj].speeds[1]; 
		cells[ii*params.nx+jj].speeds[4] = tmp_cells[ii*params.nx+jj].speeds[2]; 
		cells[ii*params.nx+jj].speeds[5] = tmp_cells[ii*params.nx+jj].speeds[7]; 
		cells[ii*params.nx+jj].speeds[6] = tmp_cells[ii*params.nx+jj].speeds[8]; 
		cells[ii*params.nx+jj].speeds[7] = tmp_cells[ii*params.nx+jj].speeds[5]; 
		cells[ii*params.nx+jj].speeds[8] = tmp_cells[ii*params.nx+jj].speeds[6]; 
	}
}

__kernel void collision(const param_t params, __global speed_t* cells, 
	      		      __global speed_t* tmp_cells, __global int* obstacles)
{
	int ii = get_global_id(0);
	int jj = get_global_id(1);
	int kk = 0;
	const double c_sq = 1.0/3.0;
	const double w0 = 4.0/9.0;
	const double w1 = 1.0/9.0;
	const double w2 = 1.0/36.0;

	double u_x, u_y;
	double u_sq;
	double local_density;
	double u[NSPEEDS];
	double d_equ[NSPEEDS];
	if(!obstacles[ii*params.nx +jj])
	{
		local_density = 0.0;
		for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += tmp_cells[ii*params.nx + jj].speeds[kk];
                }

                /* compute x velocity component */
                u_x = (tmp_cells[ii*params.nx + jj].speeds[1] +
                        tmp_cells[ii*params.nx + jj].speeds[5] +
                        tmp_cells[ii*params.nx + jj].speeds[8]
                    - (tmp_cells[ii*params.nx + jj].speeds[3] +
                        tmp_cells[ii*params.nx + jj].speeds[6] +
                        tmp_cells[ii*params.nx + jj].speeds[7]))
                    / local_density;

                /* compute y velocity component */
                u_y = (tmp_cells[ii*params.nx + jj].speeds[2] +
                        tmp_cells[ii*params.nx + jj].speeds[5] +
                        tmp_cells[ii*params.nx + jj].speeds[6]
                    - (tmp_cells[ii*params.nx + jj].speeds[4] +
                        tmp_cells[ii*params.nx + jj].speeds[7] +
                        tmp_cells[ii*params.nx + jj].speeds[8]))
                    / local_density;

                /* velocity squared */
                u_sq = u_x * u_x + u_y * u_y;

                /* directional velocity components */
                u[1] =   u_x;        /* east */
                u[2] =         u_y;  /* north */
                u[3] = - u_x;        /* west */
                u[4] =       - u_y;  /* south */
                u[5] =   u_x + u_y;  /* north-east */
                u[6] = - u_x + u_y;  /* north-west */
                u[7] = - u_x - u_y;  /* south-west */
                u[8] =   u_x - u_y;  /* south-east */

		/* equilibrium densities */
                /* zero velocity density: weight w0 */
                d_equ[0] = w0 * local_density * (1.0 - u_sq / (2.0 * c_sq));
                /* axis speeds: weight w1 */
                d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq
                    + (u[1] * u[1]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[2] = w1 * local_density * (1.0 + u[2] / c_sq
                    + (u[2] * u[2]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[3] = w1 * local_density * (1.0 + u[3] / c_sq
                    + (u[3] * u[3]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[4] = w1 * local_density * (1.0 + u[4] / c_sq
                    + (u[4] * u[4]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                /* diagonal speeds: weight w2 */
                d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq
                    + (u[5] * u[5]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[6] = w2 * local_density * (1.0 + u[6] / c_sq
                    + (u[6] * u[6]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[7] = w2 * local_density * (1.0 + u[7] / c_sq
                    + (u[7] * u[7]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));
                d_equ[8] = w2 * local_density * (1.0 + u[8] / c_sq
                    + (u[8] * u[8]) / (2.0 * c_sq * c_sq)
                    - u_sq / (2.0 * c_sq));

		    /* relaxation step */
        	    for (kk = 0; kk < NSPEEDS; kk++)
           	     {
			cells[ii*params.nx + jj].speeds[kk] =
              	       (tmp_cells[ii*params.nx + jj].speeds[kk] + params.omega *
             		      (d_equ[kk] - tmp_cells[ii*params.nx + jj].speeds[kk]));
              }
       }
}


double get_velocity(const int ii, __global speed_t * cells)
{
	int kk = 0;
	double local_density;
	double u_x;
	double u_y;

		local_density = 0.0;
		
		for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[ii].speeds[kk];
                }

                u_x = (cells[ii].speeds[1] +
                        cells[ii].speeds[5] +
                        cells[ii].speeds[8]
                    - (cells[ii].speeds[3] +
                        cells[ii].speeds[6] +
                        cells[ii].speeds[7])) /
                    local_density;

                u_y = (cells[ii].speeds[2] +
                        cells[ii].speeds[5] +
                        cells[ii].speeds[6]
                    - (cells[ii].speeds[4] +
                        cells[ii].speeds[7] +
                        cells[ii].speeds[8])) /
                    local_density;

                return sqrt(u_x*u_x + u_y*u_y);
}

__kernel void av_velocity(const param_t params, __global speed_t * cells, __global int* obstacles)
//	      			 __local double* scratch, __global double* results)
{
	int ii = get_global_id(0);

	//Initialise accumalator to zero
	double acc = 0;

	/*while (ii < params.ny*params.nx)
	{
		if(!obstacles[ii])
		{
			acc += get_velocity(ii, cells);
		}
		ii += get_global_size(0);
	}
	*/
	/*int local_index = get_local_id(0);
	

	//scratch[local_index] = cells[ii] + cells[ii+get_global_size(0)];
	barrier(CLK_LOCAL_MEM_FENCE);

	int offset = get_local_size(0)/2;
	for(;offset>1;offset/=2)
	{
		if(local_index < offset)
		{
	//		scratch[local_index] += scratch[local_index+offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(local_index==0)
	{
		results[get_group_id(0)] = scratch[0];
	}*/
}