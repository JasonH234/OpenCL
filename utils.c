/* utilities for lbm to read files, etc */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdarg.h>
#include <getopt.h>
#include <math.h>
#include "lbm.h"

void exit_with_error(int line, const char* filename, const char* format, ...)
{
    va_list arglist;

    fprintf(stderr, "Fatal error at line %d in %s: ", line, filename);

    va_start(arglist, format);
    vfprintf(stderr, format, arglist);
    va_end(arglist);

    fprintf(stderr, "\n");

    exit(EXIT_FAILURE);
}

void parse_args (int argc, char* argv[],
    char** final_state_file, char** av_vels_file, char** param_file, int * device_id)
{
    int character;

    *av_vels_file = NULL;
    *final_state_file = NULL;
    *param_file = NULL;
    *device_id = 0;

    const char * help_msg =
    "usage: ./lbm [OPTIONS] \n"
    "   -a AV_VELS_FILE\n"
    "       Name of output average velocities file\n"
    "   -f FINAL_STATE_FILE\n"
    "       Name of output final state file\n"
    "   -l \n"
    "       List OpenCL devices available\n"
    "   -d DEVICE_ID\n"
    "       Choose OpenCL device\n"
    "   -p PARAM_FILE\n"
    "       Name of input parameter file\n"
    "   -h\n"
    "       Show this message and exit\n";

    /* Used getopt to parse command line arguments for filenames */
    while ((character = getopt(argc, argv, "a:f:p:d:hl")) != -1)
    {
        switch (character)
        {
        case 'a':
            *av_vels_file = optarg;
            break;
        case 'f':
            *final_state_file = optarg;
            break;
        case 'l':
            list_opencl_platforms();
            exit(EXIT_SUCCESS);
            break;
        case 'p':
            *param_file = optarg;
            break;
        case 'd':
            sscanf(optarg, "%d", device_id);
            break;
        case 'h':
            fprintf(stderr, "%s", help_msg);
            exit(EXIT_SUCCESS);
            break;
        case '?':
            if (optopt == 'a' || optopt == 'f' || optopt == 'p' || optopt == 'd')
            {
                /* Flag present, but no option specified */
                DIE("No argument specified for '%c'", optopt);
            }
            else if (isprint(optopt))
            {
                DIE("Unknown option %c", optopt);
            }
            break;
        default:
            DIE("Error in getopt");
        }
    }

    /* Make sure they were all present */
    if (NULL == *av_vels_file)
    {
        DIE("No argument specified for av_vels file");
    }
    if (NULL == *final_state_file)
    {
        DIE("No argument specified for final state file");
    }
    if (NULL == *param_file)
    {
        DIE("No argument specified for param file");
    }
}

void initialise(const char* param_file, accel_area_t * accel_area,
		param_t* params, float** av_vels_ptr, 
		speed_t** cropped_cells, speed_t** cropped_tmp_cells, int** cropped_obstacles,
		bounds_t* bounds)
{
    FILE   *fp;            /* file pointer */
    int    ii,jj, kk;          /* generic counters */
    int    retval;         /* to hold return value for checking */
    float w0,w1,w2;       /* weighting factors */

    /* Rectangular obstacles */
    int n_obstacles;
    obstacle_t * obstacles = NULL;

    fp = fopen(param_file, "r");

    if (NULL == fp)
    {
        DIE("Unable to open param file %s", param_file);
    }

    /* read in the parameter values */
    retval = fscanf(fp,"%d\n",&(params->nx));
    if (retval != 1) DIE("Could not read param file: nx");
    retval = fscanf(fp,"%d\n",&(params->ny));
    if (retval != 1) DIE("Could not read param file: ny");
    retval = fscanf(fp,"%d\n",&(params->max_iters));
    if (retval != 1) DIE("Could not read param file: max_iters");
    retval = fscanf(fp,"%d\n",&(params->reynolds_dim));
    if (retval != 1) DIE("Could not read param file: reynolds_dim");
    retval = fscanf(fp,"%f\n",&(params->density));
    if (retval != 1) DIE("Could not read param file: density");
    retval = fscanf(fp,"%f\n",&(params->accel));
    if (retval != 1) DIE("Could not read param file: accel");
    retval = fscanf(fp,"%f\n",&(params->omega));
    if (retval != 1) DIE("Could not read param file: omega");

    if (params->nx < 100) DIE("x dimension of grid in input file was too small (must be >100)");
    if (params->ny < 100) DIE("y dimension of grid in input file was too small (must be >100)");

    /* read column/row to accelerate */
    char accel_dir_buf[11];
    int idx;
    retval = fscanf(fp,"%*s %10s %d\n", accel_dir_buf, &idx);
    if (retval != 2) DIE("Could not read param file: could not parse acceleration specification");
    if (idx > 100 || idx < 0) DIE("Acceleration index (%d) out of range (must be bigger than 0 and less than 100)", idx);

    if (!(strcmp(accel_dir_buf, "row")))
    {
        accel_area->col_or_row = ACCEL_ROW;
        accel_area->idx = idx*(params->ny/BOX_Y_SIZE);
    }
    else if (!(strcmp(accel_dir_buf, "column")))
    {
        accel_area->col_or_row = ACCEL_COLUMN;
        accel_area->idx = idx*(params->nx/BOX_X_SIZE);
    }
    else
    {
        DIE("Error reading param file: Unexpected acceleration specification '%s'", accel_dir_buf);
    }

    /* read obstacles */
    retval = fscanf(fp, "%d %*s\n", &n_obstacles);
    if (retval != 1) DIE("Could not read param file: n_obstacles");
    obstacles = (obstacle_t*) malloc(sizeof(obstacle_t)*(n_obstacles));

    for (ii = 0; ii < n_obstacles; ii++)
    {
        retval = fscanf(fp,"%f %f %f %f\n",
            &obstacles[ii].obs_x_min, &obstacles[ii].obs_y_min,
            &obstacles[ii].obs_x_max, &obstacles[ii].obs_y_max);
        if (retval != 4) DIE("Could not read param file: location of obstacle %d", ii + 1);
        if (obstacles[ii].obs_x_min < 0 || obstacles[ii].obs_y_min < 0 ||
            obstacles[ii].obs_x_max > 100 || obstacles[ii].obs_y_max > 100)
        {
            DIE("Obstacle %d out of range (must be bigger than 0 and less than 100)", ii);
        }
        if (obstacles[ii].obs_x_min > obstacles[ii].obs_x_max) DIE("Left x coordinate is bigger than right x coordinate - this will result in no obstacle being made");
        if (obstacles[ii].obs_y_min > obstacles[ii].obs_y_max) DIE("Bottom y coordinate is bigger than top y coordinate - this will result in no obstacle being made");
    }

    /* close file */
    fclose(fp);


    int *obstacles_ptr = (int*) malloc(sizeof(int)*(params->ny*params->nx));
    if (obstacles_ptr == NULL) DIE("Cannot allocate memory for patches");

    *av_vels_ptr = (float*) malloc(sizeof(float)*(params->max_iters));
    if (*av_vels_ptr == NULL) DIE("Cannot allocate memory for av_vels");


   int min_x = -1, min_y = -1, max_x = -1, max_y = -1;

    /* Fill in locations of obstacles */
    for (ii = 0; ii < params->ny; ii++)
    {
        for (jj = 0; jj < params->nx; jj++)
        {
            /* coordinates of (jj, ii) scaled to 'real world' terms */
            const float x_pos = jj*(BOX_X_SIZE/params->nx);
            const float y_pos = ii*(BOX_Y_SIZE/params->ny);
	    int is_obstacle = 0;

            for (kk = 0; kk < n_obstacles; kk++)
            {
                if (x_pos >= obstacles[kk].obs_x_min &&
                    x_pos <  obstacles[kk].obs_x_max &&
                    y_pos >= obstacles[kk].obs_y_min &&
                    y_pos <  obstacles[kk].obs_y_max)
                {
                    obstacles_ptr[ii*params->nx + jj] = 1.0;
		    is_obstacle = 1;
                }
            }
	    if(!is_obstacle)
	      {
		if( ii < min_y || min_y == -1)
		  {
		    min_y = ii;
		  }
		if (ii > max_y || max_y == -1)
		  {
		    max_y = ii;
		  }
		if (jj < min_x || min_x == -1)
		  {
		    min_x = jj;
		  }
		if (jj > max_x || max_x == -1)
		  {
		    max_x = jj;
		  }
	      }
	    else
	      {
		obstacles_ptr[ii*params->nx+jj] = 0.0;
	      }
        }
    }
    //pad
    min_x = (min_x -1 > 0) ? min_x-2 : 0;
    max_x = (max_x +1 < params->nx) ? max_x+2: params->nx;
    min_y = (min_y -1 > 0) ? min_y -2 : 0;
    max_y = (max_y +1 < params->ny) ? max_y+2 : params->ny;

    int x_size = max_x - min_x;
    int y_size = max_y - min_y;
    //y_size = pow(2, ceil(log(y_size)/log(2)));
    bounds->x = x_size;
    bounds->y = y_size;
    bounds->minx = min_x;
    bounds->maxx = max_x;
    bounds->miny = min_y;
    bounds->maxy = max_y;

    /* Allocate arrays */
    *cropped_cells = (speed_t*) malloc(sizeof(speed_t)*(x_size*y_size));
    if (*cropped_cells == NULL) DIE("Cannot allocate memory for cropped cells");

    *cropped_obstacles = (int*) malloc(sizeof(int)*(x_size*y_size));

    w0 = params->density * 4.0/9.0;
    w1 = params->density      /9.0;
    w2 = params->density      /36.0;
    printf("%d %d MINS %d %d \n", min_x, min_y, x_size, y_size);
    /* Initialise arrays */
    for (ii = 0; ii < y_size; ii++)
    {
        for (jj = 0; jj < x_size; jj++)
        {
	  
	    //centre
            (*cropped_cells)[ii*x_size + jj].speeds[0] = w0;
            // axis directions
            (*cropped_cells)[ii*x_size + jj].speeds[1] = w1;
            (*cropped_cells)[ii*x_size + jj].speeds[2] = w1;
            (*cropped_cells)[ii*x_size + jj].speeds[3] = w1;
            (*cropped_cells)[ii*x_size + jj].speeds[4] = w1;
            // diagonals
            (*cropped_cells)[ii*x_size + jj].speeds[5] = w2;
            (*cropped_cells)[ii*x_size + jj].speeds[6] = w2;
            (*cropped_cells)[ii*x_size + jj].speeds[7] = w2;
            (*cropped_cells)[ii*x_size + jj].speeds[8] = w2;
	    if(obstacles_ptr[(ii+min_y)*x_size+(jj+min_x)])
	    {
		(*cropped_obstacles)[ii*x_size+jj] = 1;
	      }
	    else 
	      {
		(*cropped_obstacles)[ii*x_size+jj]=0;
	      }
        }
    }    

    free(obstacles);
    free(obstacles_ptr);
}

void finalise(speed_t** cells_ptr, int** obstacles_ptr, float** av_vels_ptr, 
	      speed_t** cropped_cells, speed_t** cropped_tmp_cells, int** cropped_obstacles)
{
    /* Free allocated memory */
    free(*cells_ptr);
    free(*obstacles_ptr);
    free(*av_vels_ptr);
    free(*cropped_cells);
    free(*cropped_tmp_cells);
    free(*cropped_obstacles);
}

