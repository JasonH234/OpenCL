#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

/* struct to hold the parameter values */
typedef struct {
    int nx;            /* no. of cells in x-direction */
    int ny;            /* no. of cells in y-direction */
    int max_iters;      /* no. of iterations */
    int reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */
} param_t;

typedef enum { ACCEL_ROW=0, ACCEL_COLUMN=1 } accel_e;
typedef struct {
    int col_or_row;
    int idx;
} accel_area_t;


__kernel void accelerate_flow(const param_t params, const accel_area_t accel_area,
    __global float* cells, __global int* obstacles)
{
    int ii,jj;     /* generic counters */
    
    float w1,w2;  /* weighting factors */
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
            (cells[ii*params.nx + jj+4*(params.nx*params.ny)] - w1) > 0.0 &&
            (cells[ii*params.nx + jj+7*(params.nx*params.ny)] - w2) > 0.0 &&
            (cells[ii*params.nx + jj+8*(params.nx*params.ny)] - w2) > 0.0 )
            {
	    /* increase 'north-side' densities */
                cells[ii*params.nx + jj+2*(params.nx*params.ny)] += w1;
                cells[ii*params.nx + jj+5*(params.nx*params.ny)] += w2;
                cells[ii*params.nx + jj+6*(params.nx*params.ny)] += w2;
		/* decrease 'south-side' densities */
                cells[ii*params.nx + jj+4*(params.nx*params.ny)] -= w1;
                cells[ii*params.nx + jj+7*(params.nx*params.ny)] -= w2;
                cells[ii*params.nx + jj+8*(params.nx*params.ny)] -= w2;
		}
	}
	else
    	{
        ii = accel_area.idx;
	jj = get_global_id(0);

	/* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
	     (cells[ii*params.nx + jj+3*(params.nx*params.ny)] - w1) > 0.0 &&
            (cells[ii*params.nx + jj+6*(params.nx*params.ny)] - w2) > 0.0 &&
            (cells[ii*params.nx + jj+7*(params.nx*params.ny)] - w2) > 0.0 )
            {
	    /* increase 'east-side' densities */
                cells[ii*params.nx + jj+(params.nx*params.ny)] += w1;
                cells[ii*params.nx + jj+5*(params.nx*params.ny)] += w2;
                cells[ii*params.nx + jj+8*(params.nx*params.ny)] += w2;
		/* decrease 'west-side' densities */
                cells[ii*params.nx + jj+3*(params.nx*params.ny)] -= w1;
                cells[ii*params.nx + jj+6*(params.nx*params.ny)] -= w2;
                cells[ii*params.nx + jj+7*(params.nx*params.ny)] -= w2;
		}
	}
}

__kernel void propagate(const param_t params, __global float* cells, __global float* tmp_cells)
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

	    tmp_cells[ii *params.nx + jj]  = cells[ii*params.nx + jj]; /* central cell, */
                                                     /* no movement   */
            tmp_cells[ii *params.nx + x_e+(params.nx*params.ny)] = cells[ii*params.nx + jj+(params.nx*params.ny)]; /* east */
            tmp_cells[y_n*params.nx + jj+2*(params.nx*params.ny)]  = cells[ii*params.nx + jj+2*(params.nx*params.ny)]; /* north */
            tmp_cells[ii *params.nx + x_w+3*(params.nx*params.ny)] = cells[ii*params.nx + jj+3*(params.nx*params.ny)]; /* west */
            tmp_cells[y_s*params.nx + jj+4*(params.nx*params.ny)]  = cells[ii*params.nx + jj+4*(params.nx*params.ny)]; /* south */
	    tmp_cells[y_n*params.nx + x_e+5*(params.nx*params.ny)] = cells[ii*params.nx + jj+5*(params.nx*params.ny)]; /* north-east */
            tmp_cells[y_n*params.nx + x_w+6*(params.nx*params.ny)] = cells[ii*params.nx + jj+6*(params.nx*params.ny)]; /* north-west */
            tmp_cells[y_s*params.nx + x_w+7*(params.nx*params.ny)] = cells[ii*params.nx + jj+7*(params.nx*params.ny)]; /* south-west */
            tmp_cells[y_s*params.nx + x_e+8*(params.nx*params.ny)] = cells[ii*params.nx + jj+8*(params.nx*params.ny)]; /* south-east */

}

__kernel void rebound(const param_t params, __global float* cells,
	      		     __global float* tmp_cells, __global int* obstacles)
{
	int ii = get_global_id(0);
	int jj = get_global_id(1);

	if(obstacles[ii*params.nx + jj])
	{
		cells[ii*params.nx+jj+(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+3*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+2*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+4*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+3*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+4*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+2*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+5*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+7*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+6*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+8*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+7*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+5*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+8*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+6*(params.nx*params.ny)]; 
	}
}

__kernel void collision(const param_t params, __global float* cells, 
	      		      __global float* tmp_cells, __global int* obstacles)
{
	int ii = get_global_id(0);
	int jj = get_global_id(1);
	int kk = 0;
	const float c_sq = 1.0/3.0;
	const float w0 = 4.0/9.0;
	const float w1 = 1.0/9.0;
	const float w2 = 1.0/36.0;

	float u_x, u_y;
	float u_sq;
	float local_density;
	float u[NSPEEDS];
	float d_equ[NSPEEDS];

	if(obstacles[ii*params.nx + jj])
	{
		cells[ii*params.nx+jj+(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+3*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+2*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+4*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+3*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+4*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+2*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+5*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+7*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+6*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+8*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+7*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+5*(params.nx*params.ny)]; 
		cells[ii*params.nx+jj+8*(params.nx*params.ny)] = tmp_cells[ii*params.nx+jj+6*(params.nx*params.ny)]; 
	}
	if(!obstacles[ii*params.nx +jj])
	{
		local_density = 0.0;

                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += tmp_cells[ii*params.nx + jj+kk*(params.nx*params.ny)];
                }

		u_x = (tmp_cells[ii*params.nx + jj+(params.nx*params.ny)] +
                        tmp_cells[ii*params.nx + jj+5*(params.nx*params.ny)] +
                        tmp_cells[ii*params.nx + jj+8*(params.nx*params.ny)]
                    - (tmp_cells[ii*params.nx + jj+3*(params.nx*params.ny)] +
                        tmp_cells[ii*params.nx + jj+6*(params.nx*params.ny)] +
                        tmp_cells[ii*params.nx + jj+7*(params.nx*params.ny)]))
                    / local_density;

		    u_y = (tmp_cells[ii*params.nx + jj+2*(params.nx*params.ny)] +
                        tmp_cells[ii*params.nx + jj+5*(params.nx*params.ny)] +
                        tmp_cells[ii*params.nx + jj+6*(params.nx*params.ny)]
                    - (tmp_cells[ii*params.nx + jj+4*(params.nx*params.ny)] +
                        tmp_cells[ii*params.nx + jj+7*(params.nx*params.ny)] +
                        tmp_cells[ii*params.nx + jj+8*(params.nx*params.ny)]))
                    / local_density;

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
                    cells[ii*params.nx + jj+kk*(params.nx*params.ny)] = 
                        (tmp_cells[ii*params.nx + jj+kk*(params.nx*params.ny)] + params.omega * 
                        (d_equ[kk] - tmp_cells[ii*params.nx + jj+kk*(params.nx*params.ny)]));
                }

        }
}

__kernel void av_velocity(const param_t params, __global float * cells, __global int* obstacles,
	      			 __local float* scratch, __global float* results)
{

  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int global_size = get_global_size(0);
  int local_size = get_local_size(0);

  if(global_id < params.nx * params.ny)
    {
      if(!obstacles[global_id])
	{
	int kk = 0;

	float u_x, u_y;
	float local_density = 0.0;

                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[global_id+kk*(params.nx*params.ny)];
                }

		u_x = (cells[global_id+(params.nx*params.ny)] +
                        cells[global_id+5*(params.nx*params.ny)] +
                        cells[global_id+8*(params.nx*params.ny)]
                    - (cells[global_id+3*(params.nx*params.ny)] +
                        cells[global_id+6*(params.nx*params.ny)] +
                        cells[global_id+7*(params.nx*params.ny)]))
                    / local_density;

		    u_y = (cells[global_id+2*(params.nx*params.ny)] +
                        cells[global_id+5*(params.nx*params.ny)] +
                        cells[global_id+6*(params.nx*params.ny)]
                    - (cells[global_id+4*(params.nx*params.ny)] +
                        cells[global_id+7*(params.nx*params.ny)] +
                        cells[global_id+8*(params.nx*params.ny)]))
                    / local_density;

	  scratch[local_id] = sqrt(u_x * u_x + u_y * u_y);
	}
      else
	scratch[local_id] = 0;
    }

  else 
    {
      scratch[local_id] = 0;
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = 1; offset < local_size; offset <<= 1) 
    {
      int mask = (offset << 1) - 1;
      if((local_id & mask) == 0)
	{
	  scratch[local_id] = scratch[local_id] + scratch[local_id+offset];
	}
      barrier (CLK_LOCAL_MEM_FENCE);
    }

  if(local_id == 0)
    {
      results[get_group_id(0)] = scratch[0];
    }
}
